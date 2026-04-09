"""Azure Blob Storage source for dataset files.

Provides :class:`BlobSource` which lists blobs in a container via a SAS URL
and generates per-blob HTTPS URLs that can be passed directly to analysers
without any local download.

Expected SAS URL format::

    https://<account>.blob.core.windows.net/<container>?<sas-token>

Blobs are expected to be organised in virtual folders matching the local
dataset layout (e.g. ``id_cards/``, ``passports/``, …).
"""

from pathlib import PurePosixPath

from azure.storage.blob import ContainerClient

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}


class BlobSource:
    """Lists blobs in an Azure container and builds per-blob SAS URLs.

    Args:
        sas_url: Container-level SAS URL, e.g.
            ``https://<account>.blob.core.windows.net/<container>?sv=...``.
    """

    def __init__(self, sas_url: str):
        self._client = ContainerClient.from_container_url(sas_url)
        base, _, sas_token = sas_url.partition("?")
        self._base_url = base.rstrip("/")
        self._sas_token = sas_token

    def blob_url(self, blob_name: str) -> str:
        """Build the full SAS URL for a single blob."""
        return f"{self._base_url}/{blob_name}?{self._sas_token}"

    def list_files(self) -> list[tuple[str, str, str]]:
        """Return ``(file_path, filename, url)`` for every supported blob.

        ``file_path`` mirrors the local convention used in the CSV
        (e.g. ``dataset/id_cards``).  If the blob has no virtual folder the
        ``file_path`` is ``dataset``.

        Returns:
            Sorted list of ``(file_path, filename, url)`` tuples.

        Raises:
            PermissionError: If the SAS token lacks the ``List`` permission.
        """
        try:
            blobs = list(self._client.list_blobs())
        except Exception as exc:
            if "AuthorizationPermissionMismatch" in str(exc) or "not authorized" in str(exc).lower():
                raise PermissionError(
                    "BLOB_SAS_URL is missing the 'List' permission (spr=rl). "
                    "Regenerate the SAS token with at least Read + List permissions on the container."
                ) from exc
            raise

        entries = []
        for blob in blobs:
            path = PurePosixPath(blob.name)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            folder = str(path.parent) if path.parent != PurePosixPath(".") else ""
            file_path = f"dataset/{folder}" if folder else "dataset"
            entries.append((file_path, path.name, self.blob_url(blob.name)))
        return sorted(entries)
