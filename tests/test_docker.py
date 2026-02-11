"""Tests for Docker configuration files."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDockerfile:
    """Tests for Dockerfile correctness."""

    def test_dockerfile_exists(self) -> None:
        """Dockerfile should exist in project root."""
        assert (PROJECT_ROOT / "Dockerfile").exists()

    def test_dockerfile_has_multi_stage_build(self) -> None:
        """Dockerfile should use multi-stage build."""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "AS builder" in content
        assert "COPY --from=builder" in content

    def test_dockerfile_has_healthcheck(self) -> None:
        """Dockerfile should define a HEALTHCHECK."""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "HEALTHCHECK" in content

    def test_dockerfile_sets_pythonunbuffered(self) -> None:
        """Dockerfile should set PYTHONUNBUFFERED for logging."""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "PYTHONUNBUFFERED=1" in content

    def test_dockerfile_copies_src_and_configs(self) -> None:
        """Dockerfile should copy src/ and configs/ directories."""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "COPY src/ src/" in content
        assert "COPY configs/ configs/" in content

    def test_dockerfile_sets_entrypoint(self) -> None:
        """Dockerfile should set CLI as entrypoint."""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "ENTRYPOINT" in content
        assert "src.cli" in content

    def test_dockerfile_creates_directories(self) -> None:
        """Dockerfile should create required data directories."""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "mkdir -p" in content
        for dir_name in ["reports", "models", "logs", "results"]:
            assert dir_name in content


class TestDockerCompose:
    """Tests for docker-compose.yml correctness."""

    def test_docker_compose_exists(self) -> None:
        """docker-compose.yml should exist in project root."""
        assert (PROJECT_ROOT / "docker-compose.yml").exists()

    def test_docker_compose_has_service(self) -> None:
        """docker-compose.yml should define churn-predict service."""
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "churn-predict" in content

    def test_docker_compose_has_volumes(self) -> None:
        """docker-compose.yml should mount data and reports volumes."""
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "./data:/app/data" in content
        assert "./reports:/app/reports" in content

    def test_docker_compose_has_kaggle_env(self) -> None:
        """docker-compose.yml should pass Kaggle environment variables."""
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "KAGGLE_USERNAME" in content
        assert "KAGGLE_KEY" in content


class TestMakefile:
    """Tests for Makefile targets."""

    def test_makefile_exists(self) -> None:
        """Makefile should exist in project root."""
        assert (PROJECT_ROOT / "Makefile").exists()

    def test_makefile_has_docker_targets(self) -> None:
        """Makefile should have docker-build, docker-run, and docker-shell targets."""
        content = (PROJECT_ROOT / "Makefile").read_text()
        assert "docker-build:" in content
        assert "docker-run:" in content
        assert "docker-shell:" in content

    def test_makefile_has_test_target(self) -> None:
        """Makefile should have a test target with coverage."""
        content = (PROJECT_ROOT / "Makefile").read_text()
        assert "test:" in content
        assert "--cov=src" in content

    def test_makefile_has_lint_target(self) -> None:
        """Makefile should have a lint target with ruff."""
        content = (PROJECT_ROOT / "Makefile").read_text()
        assert "lint:" in content
        assert "ruff check" in content
