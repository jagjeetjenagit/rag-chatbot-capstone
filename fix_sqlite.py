"""
SQLite version workaround for ChromaDB
Replaces the system sqlite3 module with pysqlite3 if available
"""
import sys

# Try to use pysqlite3 if available (newer SQLite version)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("✓ Using pysqlite3 for newer SQLite version")
except ImportError:
    print("⚠ Using system sqlite3 - ChromaDB may not work if SQLite < 3.35.0")
    print("  To fix: Download and install newer Python version (3.9+) with newer SQLite")
