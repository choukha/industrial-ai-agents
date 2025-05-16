# Contributing to Industrial AI Agents

First off, thank you for considering contributing to the Industrial AI Agents project! Your help is essential for keeping it great.

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### 1. Report Bugs
Report bugs at [https://github.com/choukha/industrial-ai-agents/issues](https://github.com/choukha/industrial-ai-agents/issues).

If you are reporting a bug, please include:
* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.
* Expected behavior and what actually happened.
* Any error messages or stack traces.

### 2. Suggest Enhancements
Suggest enhancements at [https://github.com/choukha/industrial-ai-agents/issues](https://github.com/choukha/industrial-ai-agents/issues).

If you are suggesting an enhancement, please include:
* A clear and detailed explanation of the proposed enhancement.
* The motivation for the enhancement (what problem does it solve? what are the benefits?).
* Possible implementation ideas (optional).
* Examples of how the enhancement would be used.

### 3. Write Documentation
Industrial AI Agents could always use more documentation, whether as part of the official docs, in docstrings, or even on the web in blog posts, articles, etc.
If you want to contribute to the official documentation, please open an issue or a pull request.

### 4. Submit Feedback
The best way to send feedback is to file an issue at [https://github.com/choukha/industrial-ai-agents/issues](https://github.com/choukha/industrial-ai-agents/issues).
If you are proposing a feature:
* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Getting Started

Ready to contribute? Here's how to set up `industrial-ai-agents` for local development.

1.  **Fork the repository:**
    Click the "Fork" button at the top right of the [repository page](https://github.com/choukha/industrial-ai-agents). This will create a copy of this repository in your own GitHub account.

2.  **Clone your fork:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/industrial-ai-agents.git](https://github.com/YOUR_USERNAME/industrial-ai-agents.git)
    cd industrial-ai-agents
    ```

3.  **Create a branch:**
    Create a new branch for your local development:
    ```bash
    git checkout -b name-of-your-bugfix-or-feature
    ```
    Now you can make your changes locally.

4.  **Set up a development environment:**
    It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
    Install the project dependencies:
    ```bash
    pip install -r requirements.txt
    # Install any agent-specific dependencies if you're working on them
    pip install -r idoca/requirements.txt
    pip install -r itisa/requirements.txt
    ```
    If you plan to add new dependencies, add them to the appropriate `requirements.txt` file.

5.  **Make your changes:**
    * Write clear, concise, and well-commented code.
    * Ensure your code follows the existing style of the project (e.g., PEP 8 for Python).
    * Add tests for any new features or bug fixes, if applicable.
    * Update documentation as needed.

6.  **Commit your changes:**
    Commit your changes with a clear and descriptive commit message:
    ```bash
    git add .
    git commit -m "Your detailed description of your changes."
    ```

7.  **Push to your fork:**
    Push your changes to your forked repository on GitHub:
    ```bash
    git push origin name-of-your-bugfix-or-feature
    ```

8.  **Submit a Pull Request (PR):**
    * Go to your repository on GitHub.
    * Click the "Compare & pull request" button.
    * Provide a clear title and a detailed description of your changes in the PR.
    * Link to any relevant issues.
    * The project maintainers will review your PR. Be prepared to answer questions or make further changes if requested.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests if you are adding functionality or fixing a bug.
2.  If the pull request adds functionality, the documentation should be updated.
3.  The pull request should work for Python 3.10 and newer.
4.  Ensure your code lints (e.g., using a tool like Flake8 or Pylint, though not strictly enforced yet, aim for clean code).

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms. (You might want to add a `CODE_OF_CONDUCT.md` file separately, or include a brief statement here).

Example brief statement:
"We strive to create a welcoming and inclusive community. Please be respectful and considerate in all your interactions."

## Questions?

If you have any questions, feel free to open an issue on GitHub.

Thank you for contributing!
