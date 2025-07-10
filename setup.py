#!/usr/bin/env python3
"""
OnlineAI Package Setup Configuration
Setup script for the OnlineAI Universal AI API Gateway
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    """Read README.md for long description"""
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "OnlineAI - Universal AI API Gateway"

# Read version from app.py or set default
def get_version():
    """Get version from app.py or use default"""
    try:
        with open('app.py', 'r') as f:
            content = f.read()
            # Look for version in the health endpoint
            if '"version": "3.0"' in content:
                return "3.0.0"
    except FileNotFoundError:
        pass
    return "1.0.0"

# Core dependencies
INSTALL_REQUIRES = [
    "Flask>=3.0.0",
    "requests>=2.31.0",
    "aiohttp>=3.9.3",
]

# Optional dependencies for AI providers
EXTRAS_REQUIRE = {
    # AI Provider SDKs
    "google": ["google-generativeai>=0.3.2"],
    "openai": ["openai>=1.12.0"],
    "anthropic": ["anthropic>=0.20.0"],
    "deepseek": ["openai>=1.12.0"],  # DeepSeek uses OpenAI-compatible API
    "groq": ["groq>=0.4.2"],
    "mistral": ["mistralai>=0.4.2"],
    "xai": ["openai>=1.12.0"],  # xAI uses OpenAI-compatible API
    "perplexity": ["openai>=1.12.0"],  # Perplexity uses OpenAI-compatible API
    "meta": ["requests>=2.31.0"],  # Meta uses custom API
    "huggingface": ["transformers>=4.36.0", "torch>=2.1.0"],
    
    # Development dependencies
    "dev": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-mock>=3.12.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
    ],
    
    # Production dependencies
    "prod": [
        "gunicorn>=21.2.0",
        "uvloop>=0.19.0; sys_platform != 'win32'",
        "python-dotenv>=1.0.0",
    ],
    
    # Testing dependencies
    "test": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-mock>=3.12.0",
        "httpx>=0.26.0",
        "requests-mock>=1.11.0",
    ],
    
    # Documentation dependencies
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
    ],
}

# All AI providers
EXTRAS_REQUIRE["ai"] = (
    EXTRAS_REQUIRE["google"] +
    EXTRAS_REQUIRE["openai"] +
    EXTRAS_REQUIRE["anthropic"] +
    EXTRAS_REQUIRE["groq"] +
    EXTRAS_REQUIRE["mistral"] +
    EXTRAS_REQUIRE["huggingface"]
)

# All extras combined
EXTRAS_REQUIRE["all"] = list(set(
    sum(EXTRAS_REQUIRE.values(), [])
))

# Project URLs
PROJECT_URLS = {
    "Homepage": "https://github.com/yourusername/OnlineAI",
    "Documentation": "https://onlineai.readthedocs.io/",
    "Repository": "https://github.com/yourusername/OnlineAI",
    "Issues": "https://github.com/yourusername/OnlineAI/issues",
    "Changelog": "https://github.com/yourusername/OnlineAI/blob/main/CHANGELOG.md",
}

# Classifiers
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Framework :: Flask",
    "Topic :: Internet :: WWW/HTTP :: WSGI :: Application",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Communications :: Chat",
    "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
    "Environment :: Web Environment",
    "Typing :: Typed",
]

# Entry points
ENTRY_POINTS = {
    "console_scripts": [
        "onlineai=app:main",
        "onlineai-test=test_onlineai:main",
    ],
}

# Package data
PACKAGE_DATA = {
    "": [
        "*.html",
        "*.css",
        "*.js",
        "*.json",
        "*.md",
        "*.txt",
        "*.yml",
        "*.yaml",
    ],
}

# Data files
DATA_FILES = [
    ("", ["README.md", "requirements.txt"]),
    ("templates", ["index.html"] if os.path.exists("index.html") else []),
]

def main():
    """Main setup function"""
    setup(
        # Basic package info
        name="onlineai",
        version=get_version(),
        description="Universal AI API Gateway with intelligent fallback and streaming support",
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        
        # Author info
        author="OnlineAI Team",
        author_email="contact@onlineai.com",
        maintainer="OnlineAI Team",
        maintainer_email="contact@onlineai.com",
        
        # URLs
        url="https://github.com/yourusername/OnlineAI",
        project_urls=PROJECT_URLS,
        
        # License
        license="MIT",
        
        # Package discovery
        packages=find_packages(exclude=["tests", "tests.*"]),
        py_modules=["app", "config", "test_onlineai"],
        
        # Package data
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        include_package_data=True,
        
        # Dependencies
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        
        # Python version requirement
        python_requires=">=3.7",
        
        # Entry points
        entry_points=ENTRY_POINTS,
        
        # Classifiers
        classifiers=CLASSIFIERS,
        
        # Keywords
        keywords=[
            "ai", "api", "gateway", "chatgpt", "openai", "anthropic", "claude",
            "gemini", "google", "flask", "async", "streaming", "fallback",
            "groq", "mistral", "deepseek", "huggingface", "llama", "gpt",
            "artificial-intelligence", "machine-learning", "nlp", "chatbot",
            "text-generation", "language-model", "multi-provider", "redundancy"
        ],
        
        # Additional metadata
        platforms=["any"],
        zip_safe=False,
        
        # setuptools specific
        setup_requires=["setuptools>=45", "wheel"],
        
        # Test configuration
        test_suite="tests",
        tests_require=EXTRAS_REQUIRE["test"],
        
        # Options
        options={
            "build_scripts": {
                "executable": "/usr/bin/env python3",
            },
            "egg_info": {
                "tag_build": "",
                "tag_date": False,
            },
        },
    )

if __name__ == "__main__":
    main() 