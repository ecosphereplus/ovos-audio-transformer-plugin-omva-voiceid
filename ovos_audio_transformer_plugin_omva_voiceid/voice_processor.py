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
        self.model_cache_dir = config.get("model_cache_dir", "/tmp/models/model_cache")
        self.confidence_threshold = config.get(
            "confidence_threshold", 0.55
        )  # Lowered from 0.6 to 0.55
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
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.user_db_path), exist_ok=True)

            with open(self.user_db_path, "wb") as f:
                pickle.dump(self.user_embeddings, f)
            LOG.debug("User embeddings database saved")
        except Exception as e:
            LOG.debug(f"Failed to save user database: {e}")  # Changed to debug level

    def _prepare_audio_tensor(self, audio_data: torch.Tensor) -> torch.Tensor:
        """
        Prepare audio tensor for SpeechBrain processing with quality enhancements

        Args:
            audio_data: Raw audio tensor

        Returns:
            Processed audio tensor ready for model input
        """
        # Ensure audio is 1D
        if audio_data.dim() > 1:
            audio_data = audio_data.squeeze()

        # Basic audio preprocessing for better quality
        # Remove DC offset
        audio_data = audio_data - audio_data.mean()

        # Simple voice activity detection: remove quiet sections
        # Compute energy-based VAD with safer frame processing
        frame_size = 512
        step_size = frame_size // 2

        # Ensure we have enough audio for frame processing
        if audio_data.size(0) >= frame_size:
            # Safely unfold the audio into frames
            num_frames = (audio_data.size(0) - frame_size) // step_size + 1
            if num_frames > 0:
                frames = audio_data.unfold(0, frame_size, step_size)
                frame_energies = frames.pow(2).mean(dim=1)

                # Adaptive threshold based on audio statistics
                energy_mean = frame_energies.mean()
                energy_std = frame_energies.std()
                energy_threshold = (
                    energy_mean - 0.5 * energy_std
                )  # Keep frames above this threshold

                # Find speech regions
                speech_frames = frame_energies > energy_threshold
                if speech_frames.sum() > 0:
                    # Extract continuous speech segments
                    speech_indices = torch.nonzero(speech_frames).squeeze()
                    if speech_indices.numel() > 0:
                        if speech_indices.dim() == 0:
                            speech_indices = speech_indices.unsqueeze(0)
                        start_frame = speech_indices[0].item() * step_size
                        end_frame = min(
                            len(audio_data),
                            (speech_indices[-1].item() + 1) * step_size + frame_size,
                        )
                        audio_data = audio_data[start_frame:end_frame]

        # Ensure minimum length (3 seconds for better quality - was 1 second)
        min_length = self.sample_rate * 3  # Increased from 1 to 3 seconds
        if audio_data.size(0) < min_length:
            # Pad with zeros if too short
            padding = min_length - audio_data.size(0)
            audio_data = torch.nn.functional.pad(audio_data, (0, padding))

        # Optimal length (5-8 seconds for best results)
        optimal_length = self.sample_rate * 6  # 6 seconds
        if audio_data.size(0) > optimal_length:
            # Take the middle portion for consistency (avoid start/end artifacts)
            start_idx = (audio_data.size(0) - optimal_length) // 2
            audio_data = audio_data[start_idx : start_idx + optimal_length]

        # Normalize audio amplitude for consistent processing
        if audio_data.abs().max() > 0:
            audio_data = (
                audio_data / audio_data.abs().max() * 0.95
            )  # Leave some headroom

        # Add batch dimension if needed
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)

        return audio_data

    def retrieve_audio_tensor(self, audio_data: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Convert raw audio bytes to a PyTorch tensor.

        Args:
            audio_data: Raw audio bytes

        Returns:
            PyTorch tensor or None if conversion fails
        """
        try:
            return self._prepare_audio_tensor(audio_data)
        except Exception as e:
            LOG.error(f"Failed to convert audio data to tensor: {e}")

        return None

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
            for user_id, user_data in self.user_embeddings.items():
                # Handle both old format (numpy array) and new format (dict)
                if isinstance(user_data, dict):
                    primary_embedding = torch.from_numpy(user_data["primary"])
                    variant_embeddings = [
                        torch.from_numpy(var) for var in user_data.get("variants", [])
                    ]
                else:
                    # Legacy format compatibility
                    primary_embedding = (
                        torch.from_numpy(user_data)
                        if isinstance(user_data, np.ndarray)
                        else user_data
                    )
                    variant_embeddings = [primary_embedding]

                # Normalize query embedding
                query_norm = torch.nn.functional.normalize(
                    query_embedding.unsqueeze(0), dim=1
                )

                scores = []

                # Score against primary embedding
                primary_norm = torch.nn.functional.normalize(
                    primary_embedding.unsqueeze(0), dim=1
                )
                primary_score = self.similarity(query_norm, primary_norm)
                if isinstance(primary_score, torch.Tensor):
                    primary_score = primary_score.item()
                scores.append(primary_score)

                # Score against variant embeddings for ensemble
                for variant_embedding in variant_embeddings[:3]:  # Use top 3 variants
                    variant_norm = torch.nn.functional.normalize(
                        variant_embedding.unsqueeze(0), dim=1
                    )
                    variant_score = self.similarity(query_norm, variant_norm)
                    if isinstance(variant_score, torch.Tensor):
                        variant_score = variant_score.item()
                    scores.append(variant_score)

                # Enhanced scoring: Use best-performing approach
                if len(scores) > 1:
                    # Primary embedding gets high weight, but also consider best variant
                    primary_score = scores[0]
                    best_variant_score = max(scores[1:]) if scores[1:] else 0

                    # Weight primary heavily but allow best variant to boost
                    final_score = 0.7 * primary_score + 0.3 * best_variant_score

                    # Bonus if any variant score is very close to primary (consistency)
                    variant_scores = scores[1:]
                    if any(abs(vs - primary_score) < 0.1 for vs in variant_scores):
                        final_score += 0.05  # Small consistency bonus
                else:
                    final_score = scores[0]

                LOG.debug(
                    f"User {user_id}: primary={primary_score:.3f}, final={final_score:.3f}"
                )

                if final_score > best_score:
                    best_score = final_score
                    best_user = user_id

            # Check if best score meets threshold with adaptive calibration
            calibrated_score = self._calibrate_confidence(
                best_score, len(self.user_embeddings)
            )

            if calibrated_score >= self.confidence_threshold:
                LOG.info(
                    f"Speaker identified: {best_user} (raw: {best_score:.3f}, calibrated: {calibrated_score:.3f})"
                )
                return best_user, calibrated_score
            else:
                LOG.debug(
                    f"No confident match found (raw: {best_score:.3f}, calibrated: {calibrated_score:.3f} < {self.confidence_threshold})"
                )
                return None, calibrated_score

        except Exception as e:
            LOG.error(f"Speaker identification failed: {e}")
            return None, 0.0

    def _calibrate_confidence(self, raw_score: float, num_enrolled_users: int) -> float:
        """
        Calibrate confidence scores for better thresholding

        Args:
            raw_score: Raw similarity score
            num_enrolled_users: Number of enrolled users

        Returns:
            Calibrated confidence score
        """
        # Base calibration: Apply sigmoid-like scaling to spread scores
        calibrated = raw_score

        # Multi-user penalty: As more users are enrolled, be more conservative
        if num_enrolled_users > 1:
            user_penalty = min(
                0.05, (num_enrolled_users - 1) * 0.015
            )  # Reduced penalty
            calibrated -= user_penalty

        # More balanced thresholding
        if raw_score > 0.75:
            # Very high confidence boost
            calibrated += min(0.2, (raw_score - 0.75) * 1.2)
        elif raw_score > 0.6:
            # High confidence moderate boost
            calibrated += min(0.15, (raw_score - 0.6) * 0.8)
        elif raw_score > 0.4:
            # Medium confidence slight boost
            calibrated += min(0.1, (raw_score - 0.4) * 0.5)
        elif raw_score < 0.25:
            # Very low confidence penalty
            calibrated *= 0.3
        else:
            # Low confidence mild penalty
            calibrated *= 0.7

        # Non-linear scaling to improve separation
        # Apply sigmoid with gentler scaling
        import math

        sigmoid_factor = 2.0  # Gentler steepness
        calibrated = 1.0 / (
            1.0 + math.exp(-sigmoid_factor * (calibrated - 0.5))
        )  # Center at 0.5

        # Ensure reasonable bounds
        calibrated = max(0.0, min(1.0, calibrated))

        return calibrated

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

            # Enhanced enrollment: Store multiple embeddings instead of just averaging
            # This allows for better matching against variations in speech
            if len(embeddings) >= 3:
                # Use multiple representative embeddings for better coverage
                embedding_stack = torch.stack(embeddings)

                # Store the centroid (average) as primary
                avg_embedding = embedding_stack.mean(dim=0)

                # Also store individual embeddings for ensemble matching
                self.user_embeddings[user_id] = {
                    "primary": avg_embedding.cpu().numpy(),
                    "variants": [
                        emb.cpu().numpy() for emb in embeddings[:5]
                    ],  # Store up to 5 variants
                }
            else:
                # Fallback for fewer samples
                if len(embeddings) > 1:
                    avg_embedding = torch.stack(embeddings).mean(dim=0)
                else:
                    avg_embedding = embeddings[0]

                self.user_embeddings[user_id] = {
                    "primary": avg_embedding.cpu().numpy(),
                    "variants": [emb.cpu().numpy() for emb in embeddings],
                }

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

            # Convert to scalar if tensor
            if isinstance(score, torch.Tensor):
                score = score.item()

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
            LOG.debug(
                f"Failed to save user database during cleanup: {e}"
            )  # Changed to debug level

        # Clear model to free memory
        self.verification_model = None

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except Exception:
            pass  # Suppress all cleanup errors during destruction
