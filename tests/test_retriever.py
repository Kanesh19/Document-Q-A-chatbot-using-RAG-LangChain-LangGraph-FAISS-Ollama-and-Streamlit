from pathlib import Path

from src.config import DATA_DIR, VECTORSTORE_DIR


def test_project_directories_exist():
    assert DATA_DIR.exists()
    assert VECTORSTORE_DIR.exists()


def test_supported_input_directory_is_project_local():
    assert Path(DATA_DIR).name == "documents"
    assert Path(VECTORSTORE_DIR).name == "faiss_index"
