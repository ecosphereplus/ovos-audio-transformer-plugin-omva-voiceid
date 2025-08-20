"""
OMVA Voice Processor using SpeechBrain

This module implements voice identification using SpeechBrain's pre-trained models
for speaker recognition and verification.
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ovos_utils.log import LOG
from speechbrain.inference.speaker import SpeakerRecognition


class OMVAVoiceProcessor:
    """
    OMVA Voice Processor using SpeechBrain ECAPA-TDNN model

    Provides speaker identification and verification capabilities using
    state-of-the-art pre-trained models from SpeechBrain.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the voice processor

        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config

        # Model configuration
        self.model_source = config.get(
            "model_source", "speechbrain/spkrec-ecapa-voxceleb"
        )
        self.model_cache_dir = config.get(
            "model_cache_dir", "./models/speechbrain_cache"
        )
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        self.sample_rate = config.get("sample_rate", 16000)
        self.gpu = config.get("gpu", False)

        # User database
        self.user_embeddings = {}  # user_id -> embedding
        self.user_db_path = os.path.join(self.model_cache_dir, "user_embeddings.pkl")

        # Ensure cache directory exists
        os.makedirs(self.model_cache_dir, exist_ok=True)

        # Initialize SpeechBrain model
        self.verification_model = None
        self._initialize_model()

        # Load existing user database
        self._load_user_database()

        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        LOG.info(
            f"OMVA Voice Processor initialized with {len(self.user_embeddings)} enrolled users"
        )

    def _initialize_model(self):
        """Initialize SpeechBrain speaker verification model"""

        try:
            LOG.info(f"Loading SpeechBrain model: {self.model_source}")
            opts = {"device": "cpu"}
            if self.gpu and torch.cuda.is_available():
                opts["device"] = "cuda"
            self.verification_model = SpeakerRecognition.from_hparams(
                source=self.model_source,
                savedir=self.model_cache_dir,
                run_opts=opts,
            )
            LOG.info("SpeechBrain model loaded successfully")

        except Exception as e:
            LOG.error(f"Failed to load SpeechBrain model: {e}")
            self.verification_model = None

    def _load_user_database(self):
        """Load user embeddings database from disk"""
        if os.path.exists(self.user_db_path):
            try:
                with open(self.user_db_path, "rb") as f:
                    self.user_embeddings = pickle.load(f)
                LOG.info(
                    f"Loaded {len(self.user_embeddings)} user embeddings from database"
                )
            except Exception as e:
                LOG.error(f"Failed to load user database: {e}")
                self.user_embeddings = {}

    def _save_user_database(self):
        """Save user embeddings database to disk"""
        try:
            with open(self.user_db_path, "wb") as f:
                pickle.dump(self.user_embeddings, f)
            LOG.debug("User embeddings database saved")
        except Exception as e:
            LOG.error(f"Failed to save user database: {e}")

    def _prepare_audio_tensor(self, audio_data: torch.Tensor) -> torch.Tensor:
        """
        Prepare audio tensor for SpeechBrain processing

        Args:
            audio_data: Raw audio tensor

        Returns:
            Processed audio tensor ready for model input
        """
        # Ensure audio is 1D
        if audio_data.dim() > 1:
            audio_data = audio_data.squeeze()

        # Ensure minimum length (1 second)
        min_length = self.sample_rate
        if audio_data.size(0) < min_length:
            # Pad with zeros if too short
            padding = min_length - audio_data.size(0)
            audio_data = torch.nn.functional.pad(audio_data, (0, padding))

        # Limit maximum length (10 seconds for memory efficiency)
        max_length = self.sample_rate * 10
        if audio_data.size(0) > max_length:
            audio_data = audio_data[:max_length]

        # Add batch dimension if needed
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)

        return audio_data

    def extract_embedding(self, audio_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract speaker embedding from audio using SpeechBrain

        Args:
            audio_tensor: Input audio tensor

        Returns:
            Speaker embedding tensor or None if extraction fails
        """
        if self.verification_model is None:
            LOG.warning("SpeechBrain model not available")
            return None

        try:
            # Prepare audio for processing
            processed_audio = self._prepare_audio_tensor(audio_tensor)

            # Extract embedding using SpeechBrain
            embedding = self.verification_model.encode_batch(processed_audio)

            # Return embedding as CPU tensor
            return embedding.squeeze().cpu()

        except Exception as e:
            LOG.error(f"Failed to extract embedding: {e}")
            return None

    def identify_speaker(
        self, audio_tensor: torch.Tensor
    ) -> Tuple[Optional[str], float]:
        """
        Identify speaker from audio tensor

        Args:
            audio_tensor: Input audio tensor

        Returns:
            Tuple of (speaker_id, confidence) or (None, 0.0) if no match
        """
        if not self.user_embeddings:
            LOG.debug("No users enrolled for identification")
            return None, 0.0

        # Extract embedding from input audio
        query_embedding = self.extract_embedding(audio_tensor)
        if query_embedding is None:
            return None, 0.0

        # Compare with all enrolled users
        best_user = None
        best_score = 0.0

        try:
            for user_id, stored_embedding in self.user_embeddings.items():
                # Compute cosine similarity
                if isinstance(stored_embedding, np.ndarray):
                    stored_embedding = torch.from_numpy(stored_embedding)

                # Normalize embeddings for cosine similarity
                query_norm = torch.nn.functional.normalize(
                    query_embedding.unsqueeze(0), dim=1
                )
                stored_norm = torch.nn.functional.normalize(
                    stored_embedding.unsqueeze(0), dim=1
                )

                # Compute similarity score
                # similarity = torch.nn.functional.cosine_similarity(
                #     query_norm, stored_norm
                # ).item()

                similarity = self.similarity(query_norm, stored_norm)

                LOG.debug(f"Similarity with {user_id}: {similarity:.3f}")

                if similarity > best_score:
                    best_score = similarity
                    best_user = user_id

            # Check if best score meets threshold
            if best_score >= self.confidence_threshold:
                LOG.info(
                    f"Speaker identified: {best_user} (confidence: {best_score:.3f})"
                )
                return best_user, best_score
            else:
                LOG.debug(
                    f"No confident match found (best: {best_score:.3f} < {self.confidence_threshold})"
                )
                return None, best_score

        except Exception as e:
            LOG.error(f"Speaker identification failed: {e}")
            return None, 0.0

    def enroll_user(self, user_id: str, audio_samples: List[torch.Tensor]) -> bool:
        """
        Enroll a new user with voice samples

        Args:
            user_id: Unique identifier for the user
            audio_samples: List of audio tensors for training

        Returns:
            True if enrollment successful, False otherwise
        """
        if not audio_samples:
            LOG.error("No audio samples provided for enrollment")
            return False

        try:
            LOG.info(f"Enrolling user: {user_id} with {len(audio_samples)} samples")

            # Extract embeddings from all samples
            embeddings = []
            for i, audio in enumerate(audio_samples):
                embedding = self.extract_embedding(audio)
                if embedding is not None:
                    embeddings.append(embedding)
                    LOG.debug(f"Extracted embedding {i+1}/{len(audio_samples)}")
                else:
                    LOG.warning(f"Failed to extract embedding from sample {i+1}")

            if not embeddings:
                LOG.error("No valid embeddings extracted from samples")
                return False

            # Average multiple embeddings for robustness
            if len(embeddings) > 1:
                avg_embedding = torch.stack(embeddings).mean(dim=0)
            else:
                avg_embedding = embeddings[0]

            # Store user embedding
            self.user_embeddings[user_id] = avg_embedding.cpu().numpy()

            # Save to database
            self._save_user_database()

            LOG.info(
                f"User {user_id} enrolled successfully with {len(embeddings)} valid samples"
            )
            return True

        except Exception as e:
            LOG.error(f"User enrollment failed for {user_id}: {e}")
            return False

    def get_enrolled_users(self) -> List[str]:
        """Get list of enrolled user IDs"""
        return list(self.user_embeddings.keys())

    def remove_user(self, user_id: str) -> bool:
        """
        Remove an enrolled user

        Args:
            user_id: User ID to remove

        Returns:
            True if removal successful, False otherwise
        """
        try:
            if user_id in self.user_embeddings:
                del self.user_embeddings[user_id]
                self._save_user_database()
                LOG.info(f"User {user_id} removed successfully")
                return True
            else:
                LOG.warning(f"User {user_id} not found in enrolled users")
                return False
        except Exception as e:
            LOG.error(f"Failed to remove user {user_id}: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model and enrolled users"""
        return {
            "model_source": self.model_source,
            "model_available": self.verification_model is not None,
            "enrolled_users": len(self.user_embeddings),
            "users": list(self.user_embeddings.keys()),
            "confidence_threshold": self.confidence_threshold,
            "sample_rate": self.sample_rate,
            "model_cache_dir": self.model_cache_dir,
        }

    def verify_speakers(
        self, audio1: torch.Tensor, audio2: torch.Tensor
    ) -> Tuple[bool, float]:
        """
        Verify if two audio samples are from the same speaker

        Args:
            audio1: First audio tensor
            audio2: Second audio tensor

        Returns:
            Tuple of (is_same_speaker, similarity_score)
        """
        try:
            # Extract embeddings from both audio samples
            emb1 = self.extract_embedding(audio1)
            emb2 = self.extract_embedding(audio2)

            if emb1 is None or emb2 is None:
                return False, 0.0

            # Compute cosine similarity
            emb1_norm = torch.nn.functional.normalize(emb1.unsqueeze(0), dim=1)
            emb2_norm = torch.nn.functional.normalize(emb2.unsqueeze(0), dim=1)

            # similarity = torch.nn.functional.cosine_similarity(
            #     emb1_norm, emb2_norm
            # ).item()

            score = self.similarity(emb1_norm, emb2_norm)

            is_same = score >= self.confidence_threshold

            LOG.debug(
                f"Speaker verification: similarity={score:.3f}, same_speaker={is_same}"
            )

            return is_same, score

        except Exception as e:
            LOG.error(f"Speaker verification failed: {e}")
            return False, 0.0

    def cleanup(self):
        """Cleanup resources"""
        LOG.debug("Cleaning up voice processor resources")

        # Save user database one final time
        try:
            self._save_user_database()
        except Exception as e:
            LOG.error(f"Failed to save user database during cleanup: {e}")

        # Clear model to free memory
        self.verification_model = None

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except Exception:
            pass
