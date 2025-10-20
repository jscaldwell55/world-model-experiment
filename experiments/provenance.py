# experiments/provenance.py
import hashlib
import subprocess
import inspect
from datetime import datetime
from typing import Any
import json

def get_git_sha() -> str:
    """Get current git commit SHA"""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return "unknown"

def get_git_diff() -> str:
    """Get uncommitted changes"""
    try:
        return subprocess.check_output(
            ['git', 'diff', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode()
    except:
        return ""

def hash_module(module) -> str:
    """SHA-256 hash of module source code"""
    try:
        source = inspect.getsource(module)
        return hashlib.sha256(source.encode()).hexdigest()
    except:
        return "unknown"

def hash_file(filepath: str) -> str:
    """SHA-256 hash of file contents"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return "unknown"

class ProvenanceLog:
    """Complete provenance for reproducibility"""
    
    def __init__(self, config: dict, environment_module: Any, agent_module: Any):
        self.timestamp = datetime.utcnow().isoformat()
        self.config = config
        self.code_sha = get_git_sha()
        self.code_diff = get_git_diff()
        self.environment_version = hash_module(environment_module)
        self.agent_version = hash_module(agent_module)
        self.config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'config': self.config,
            'code_sha': self.code_sha,
            'has_uncommitted_changes': len(self.code_diff) > 0,
            'environment_version': self.environment_version,
            'agent_version': self.agent_version,
            'config_hash': self.config_hash,
        }
    
    def verify(self, logged_provenance: dict) -> bool:
        """Verify a logged episode matches current code"""
        return (
            self.environment_version == logged_provenance['environment_version'] and
            self.agent_version == logged_provenance['agent_version']
        )