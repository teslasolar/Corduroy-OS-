"""
KONOMI Core - Konomi Orchestrator Module
Main orchestrator for the entire Corduroy-OS system

Re-exports Konomi class for backward compatibility.
"""

from .core import Konomi

# Convenience instance for quick imports
konomi = Konomi()

__all__ = ['Konomi', 'konomi']
