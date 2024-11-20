#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# Add the root directory (where src is located) to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))




def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'search_engine.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()