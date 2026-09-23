import urllib.error
import zipfile

from scripts.download_luna16 import _extract_subset, _is_transient_download_error


def test_downloader_retries_temporary_gateway_errors():
    temporary = urllib.error.HTTPError("https://example.invalid", 504, "timeout", {}, None)
    permanent = urllib.error.HTTPError("https://example.invalid", 404, "missing", {}, None)

    assert _is_transient_download_error(temporary)
    assert not _is_transient_download_error(permanent)


def test_extraction_resumes_an_incomplete_generated_staging_directory(tmp_path):
    archive = tmp_path / "subset0.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("subset0/scan.mhd", "complete header")
        zip_file.writestr("subset0/scan.raw", b"complete pixels")

    staging_subset = tmp_path / ".subset0.extracting" / "subset0"
    staging_subset.mkdir(parents=True)
    (staging_subset / "scan.mhd").write_text("partial", encoding="utf-8")

    _extract_subset(archive, tmp_path, 0)

    assert (tmp_path / "subset0" / "scan.mhd").read_text(encoding="utf-8") == "complete header"
    assert (tmp_path / "subset0" / "scan.raw").read_bytes() == b"complete pixels"
    assert not (tmp_path / ".subset0.extracting").exists()
