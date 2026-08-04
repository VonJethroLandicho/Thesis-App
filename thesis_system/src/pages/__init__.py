"""Streamlit page package.

Page modules are imported by :mod:`app` only when the user opens that page.
Keeping this package initializer import-free avoids loading every page and its
backend dependencies during application startup.
"""
