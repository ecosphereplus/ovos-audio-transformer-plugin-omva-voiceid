#!/usr/bin/env python3
import os
from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def get_version():
    version_file = os.path.join(
        os.path.dirname(__file__),
        "ovos_audio_transformer_plugin_omva_voiceid",
        "version.py",
    )
    version = "0.0.1"  # Default version if version file does not exist
    if os.path.exists(version_file):
        major, minor, build, alpha = "0", "0", "1", "0"
        with open(version_file, encoding="utf-8") as f:
            for line in f:
                if "VERSION_MAJOR" in line:
                    major = line.split("=")[1].strip()
                elif "VERSION_MINOR" in line:
                    minor = line.split("=")[1].strip()
                elif "VERSION_BUILD" in line:
                    build = line.split("=")[1].strip()
        version = f"{major}.{minor}.{build}"
        if alpha != "0":
            version += f"-alpha.{alpha}"
    return version


def package_files(directory):
    paths = []
    for path, _, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


def required(requirements_file):
    """Read requirements file and remove comments and empty lines."""
    with open(os.path.join(BASEDIR, requirements_file), "r", encoding="utf-8") as f:
        requirements = f.read().splitlines()
        if "MYCROFT_LOOSE_REQUIREMENTS" in os.environ:
            print("USING LOOSE REQUIREMENTS!")
            requirements = [
                r.replace("==", ">=").replace("~=", ">=") for r in requirements
            ]
        return [pkg for pkg in requirements if pkg.strip() and not pkg.startswith("#")]


PLUGIN_ENTRY_POINT = (
    "ovos-audio-transformer-plugin-omva-voiceid = "
    "ovos_audio_transformer_plugin_omva_voiceid:OMVAVoiceIDPlugin"
)

CONFIG_ENTRY_POINT = (
    "ovos-audio-transformer-plugin-omva-voiceid.config = "
    "ovos_audio_transformer_plugin_omva_voiceid:OMVAVoiceIDConfig"
)

setup(
    name="ovos-audio-transformer-plugin-omva-voiceid",
    version=get_version(),
    description="OMVA Voice Identification Plugin for OVOS",
    long_description=(
        "A plugin for the Open Voice OS (OVOS) that provides voice "
        "identification capabilities using the OMVA model."
    ),
    author="OMVA Team",
    url="https://github.com/ecosphereplus/ovos-audio-transformer-plugin-omva-voiceid",
    author_email="contact@omva.ai",
    packages=["ovos_audio_transformer_plugin_omva_voiceid"],
    package_data={"": package_files("ovos_audio_transformer_plugin_omva_voiceid")},
    install_requires=required("requirements.txt"),
    entry_points={
        "opm.transformer.audio": [PLUGIN_ENTRY_POINT],
        "opm.transformer.audio.config": [CONFIG_ENTRY_POINT],
        "console_scripts": [
            "ovos-omva-voiceid-listener=ovos_audio_transformer_plugin_omva_voiceid:launch_cli"
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="ovos plugin audio transformer omva voiceid",
    zip_safe=True,
)
