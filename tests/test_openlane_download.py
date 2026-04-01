from __future__ import annotations

from tsqbev.openlane_download import (
    _detect_html_error,
    _DriveFormParser,
    openlanev2_archives,
    resolve_archive_keys,
)


def test_drive_form_parser_extracts_hidden_inputs() -> None:
    parser = _DriveFormParser()
    parser.feed(
        """
        <html><body>
          <form>
            <input type="hidden" name="id" value="abc123"/>
            <input type="hidden" name="export" value="download"/>
            <input type="hidden" name="confirm" value="t"/>
            <input type="hidden" name="uuid" value="uuid-123"/>
          </form>
        </body></html>
        """
    )

    assert parser.inputs["id"] == "abc123"
    assert parser.inputs["confirm"] == "t"
    assert parser.inputs["uuid"] == "uuid-123"


def test_openlanev2_archives_expose_sample_manifest() -> None:
    archives = {entry["key"]: entry for entry in openlanev2_archives()}

    assert "sample" in archives
    assert archives["sample"]["filename"] == "OpenLane-V2_sample.tar"
    assert archives["sample"]["md5"] == "21c607fa5a1930275b7f1409b25042a0"


def test_resolve_archive_keys_defaults_to_sample() -> None:
    assert resolve_archive_keys(None) == ["sample"]
    assert resolve_archive_keys([]) == ["sample"]
    assert resolve_archive_keys(["subset_a_info", "sample"]) == ["subset_a_info", "sample"]


def test_detect_html_error_classifies_google_drive_quota(tmp_path) -> None:
    path = tmp_path / "quota.html"
    path.write_text("<html><title>Google Drive - Quota exceeded</title></html>")

    assert _detect_html_error(path) == "google_drive_quota_exceeded"
