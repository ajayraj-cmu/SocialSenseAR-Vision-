"""
Core module for Quest Processor.

Contains the main pipeline and transition management.
"""
from .transition import TransitionEffect
from .pipeline import Pipeline

__all__ = ['TransitionEffect', 'Pipeline']
