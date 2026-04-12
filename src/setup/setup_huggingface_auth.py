#!/usr/bin/env python3
"""
Helper script to set up HuggingFace authentication.
This stores your HuggingFace token locally so datasets can be accessed.
"""

from huggingface_hub import login
from pathlib import Path
import os

def setup_authentication():
    """
    Prompt user for HuggingFace token and save it locally.
    """
    print("\n" + "="*80)
    print("HUGGINGFACE AUTHENTICATION SETUP")
    print("="*80)
    print()
    print("To access gated datasets, you need a HuggingFace token.")
    print()
    print("Steps:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token (if you don't have one)")
    print("3. Copy the token value")
    print("4. Paste it below when prompted")
    print()
    
    token = input("Enter your HuggingFace token: ").strip()
    
    if not token:
        print("ERROR: No token provided.")
        return False
    
    try:
        # Save token locally
        login(token=token, add_to_git_credential=True)
        print("\n✓ Token saved successfully!")
        print("You can now access gated datasets.")
        
        # Verify token is saved
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            print(f"✓ Token file created at: {token_path}")
        
        return True
    except Exception as e:
        print(f"\nERROR: Failed to save token: {e}")
        return False

if __name__ == "__main__":
    success = setup_authentication()
    exit(0 if success else 1)
