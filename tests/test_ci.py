"""Tests for GitHub Actions CI/CD configuration."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CI_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


class TestCIWorkflow:
    """Tests for the CI workflow configuration."""

    def test_ci_workflow_exists(self) -> None:
        """CI workflow file should exist."""
        assert CI_PATH.exists()

    def test_ci_workflow_is_valid_yaml(self) -> None:
        """CI workflow should be valid YAML."""
        content = yaml.safe_load(CI_PATH.read_text())
        assert isinstance(content, dict)

    def test_ci_workflow_name(self) -> None:
        """CI workflow should have a name."""
        content = yaml.safe_load(CI_PATH.read_text())
        assert content["name"] == "CI"

    def test_ci_triggers_on_push_and_pr(self) -> None:
        """CI should trigger on push to main and pull requests."""
        content = yaml.safe_load(CI_PATH.read_text())
        # PyYAML converts 'on' key to boolean True
        triggers = content[True]
        assert "push" in triggers
        assert "pull_request" in triggers
        assert "main" in triggers["push"]["branches"]
        assert "main" in triggers["pull_request"]["branches"]

    def test_ci_has_lint_job(self) -> None:
        """CI should have a lint job with ruff."""
        content = yaml.safe_load(CI_PATH.read_text())
        assert "lint" in content["jobs"]
        lint_steps = content["jobs"]["lint"]["steps"]
        step_names = [s.get("name", "") for s in lint_steps]
        assert any("ruff check" in n.lower() for n in step_names)
        assert any("ruff format" in n.lower() for n in step_names)

    def test_ci_has_test_job(self) -> None:
        """CI should have a test job with pytest and coverage."""
        content = yaml.safe_load(CI_PATH.read_text())
        assert "test" in content["jobs"]
        test_steps = content["jobs"]["test"]["steps"]
        step_runs = [s.get("run", "") for s in test_steps]
        assert any("pytest" in r for r in step_runs)
        assert any("coverage" in r for r in step_runs)

    def test_ci_test_depends_on_lint(self) -> None:
        """Test job should depend on lint job."""
        content = yaml.safe_load(CI_PATH.read_text())
        assert content["jobs"]["test"]["needs"] == "lint"

    def test_ci_has_docker_job(self) -> None:
        """CI should have a docker build job."""
        content = yaml.safe_load(CI_PATH.read_text())
        assert "docker" in content["jobs"]
        docker_steps = content["jobs"]["docker"]["steps"]
        step_runs = [s.get("run", "") for s in docker_steps]
        assert any("docker build" in r for r in step_runs)

    def test_ci_docker_depends_on_test(self) -> None:
        """Docker job should depend on test job."""
        content = yaml.safe_load(CI_PATH.read_text())
        assert content["jobs"]["docker"]["needs"] == "test"

    def test_ci_uses_python_311(self) -> None:
        """CI should use Python 3.11."""
        content = yaml.safe_load(CI_PATH.read_text())
        test_steps = content["jobs"]["test"]["steps"]
        python_step = next(s for s in test_steps if s.get("name") == "Set up Python")
        assert python_step["with"]["python-version"] == "3.11"

    def test_ci_coverage_threshold(self) -> None:
        """CI should enforce 80% coverage threshold."""
        content = yaml.safe_load(CI_PATH.read_text())
        test_steps = content["jobs"]["test"]["steps"]
        step_runs = [s.get("run", "") for s in test_steps]
        assert any("--fail-under=80" in r for r in step_runs)
