#!/usr/bin/env python3
"""
Font configuration utility for matplotlib Chinese character support
"""

import os
import sys
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings


def configure_matplotlib_fonts():
    """
    Configure matplotlib to handle Chinese fonts properly
    """
    # Suppress font warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # Try to find available Chinese fonts
    chinese_fonts = find_chinese_fonts()
    
    if chinese_fonts:
        # Use the first available Chinese font
        matplotlib.rcParams['font.sans-serif'] = [chinese_fonts[0]] + matplotlib.rcParams['font.sans-serif']
        print(f"Using Chinese font: {chinese_fonts[0]}")
    else:
        # Fallback to DejaVu Sans and use English labels
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("No Chinese fonts found, using English labels")
    
    # Handle minus sign display
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # Clear font cache
    fm._rebuild()


def find_chinese_fonts():
    """
    Find available Chinese fonts on the system
    
    Returns:
        list: List of available Chinese font names
    """
    chinese_fonts = []
    
    # Common Chinese font names to look for
    chinese_font_patterns = [
        'SimHei', 'SimSun', 'Microsoft YaHei', 'WenQuanYi',
        'Noto Sans CJK', 'Source Han Sans', 'Droid Sans Fallback',
        'AR PL UMing', 'AR PL UKai', 'WenQuanYi Micro Hei'
    ]
    
    # Get all available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Check for Chinese fonts
    for pattern in chinese_font_patterns:
        for font in available_fonts:
            if pattern.lower() in font.lower():
                chinese_fonts.append(font)
                break
    
    return list(set(chinese_fonts))  # Remove duplicates


def install_chinese_fonts():
    """
    Install Chinese fonts on the system (requires sudo privileges)
    """
    print("Attempting to install Chinese fonts...")
    
    # Detect the operating system
    if sys.platform.startswith('linux'):
        install_linux_chinese_fonts()
    elif sys.platform == 'darwin':
        install_macos_chinese_fonts()
    elif sys.platform.startswith('win'):
        print("On Windows, Chinese fonts are usually pre-installed.")
        print("If you're having issues, try installing 'Microsoft YaHei' or 'SimHei'")
    else:
        print(f"Unsupported operating system: {sys.platform}")


def install_linux_chinese_fonts():
    """
    Install Chinese fonts on Linux systems
    """
    try:
        # Try different package managers
        if subprocess.run(['which', 'apt'], capture_output=True).returncode == 0:
            # Debian/Ubuntu
            subprocess.run([
                'sudo', 'apt', 'update'
            ], check=True)
            subprocess.run([
                'sudo', 'apt', 'install', '-y', 
                'fonts-wqy-microhei', 'fonts-wqy-zenhei', 
                'fonts-noto-cjk', 'fonts-arphic-uming'
            ], check=True)
        elif subprocess.run(['which', 'yum'], capture_output=True).returncode == 0:
            # CentOS/RHEL
            subprocess.run([
                'sudo', 'yum', 'install', '-y',
                'wqy-microhei-fonts', 'wqy-zenhei-fonts'
            ], check=True)
        elif subprocess.run(['which', 'pacman'], capture_output=True).returncode == 0:
            # Arch Linux
            subprocess.run([
                'sudo', 'pacman', '-S', '--noconfirm',
                'wqy-microhei', 'wqy-zenhei', 'noto-fonts-cjk'
            ], check=True)
        else:
            print("Could not detect package manager. Please install Chinese fonts manually.")
            return False
            
        print("Chinese fonts installed successfully!")
        print("Please restart your Python session for changes to take effect.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Chinese fonts: {e}")
        return False
    except FileNotFoundError:
        print("Package manager not found. Please install Chinese fonts manually.")
        return False


def install_macos_chinese_fonts():
    """
    Install Chinese fonts on macOS
    """
    try:
        # Check if Homebrew is available
        if subprocess.run(['which', 'brew'], capture_output=True).returncode == 0:
            subprocess.run([
                'brew', 'install', '--cask', 'font-source-han-sans'
            ], check=True)
            print("Chinese fonts installed successfully!")
            return True
        else:
            print("Homebrew not found. Please install Chinese fonts manually.")
            print("You can download fonts from: https://github.com/adobe-fonts/source-han-sans")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Chinese fonts: {e}")
        return False


def test_chinese_display():
    """
    Test if Chinese characters can be displayed properly
    """
    configure_matplotlib_fonts()
    
    # Create a simple test plot
    plt.figure(figsize=(8, 6))
    plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
    
    # Try to use Chinese characters
    try:
        plt.title('测试中文显示 - Test Chinese Display')
        plt.xlabel('横轴 - X Axis')
        plt.ylabel('纵轴 - Y Axis')
        chinese_support = True
    except:
        # Fallback to English
        plt.title('Test Chinese Display - English Fallback')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        chinese_support = False
    
    plt.grid(True, alpha=0.3)
    plt.savefig('font_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    if chinese_support:
        print("Chinese font test completed. Check 'font_test.png' for results.")
    else:
        print("Chinese fonts not available, using English fallback.")
    
    return chinese_support


def main():
    """
    Main function to handle font configuration
    """
    print("Matplotlib Chinese Font Configuration Utility")
    print("=" * 50)
    
    # Check current font status
    chinese_fonts = find_chinese_fonts()
    print(f"Available Chinese fonts: {chinese_fonts if chinese_fonts else 'None'}")
    
    if not chinese_fonts:
        print("\nNo Chinese fonts detected.")
        choice = input("Would you like to install Chinese fonts? (y/n): ").lower().strip()
        
        if choice == 'y':
            success = install_chinese_fonts()
            if success:
                # Re-check after installation
                chinese_fonts = find_chinese_fonts()
    
    # Configure matplotlib
    configure_matplotlib_fonts()
    
    # Test display
    print("\nTesting font display...")
    test_chinese_display()
    
    print("\nFont configuration completed!")


if __name__ == "__main__":
    main() 