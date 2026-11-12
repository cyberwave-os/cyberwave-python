"""REST CA selection: use the OS store when it works, else a host bundle, else certifi."""

import ssl

import certifi
import pytest

from cyberwave import client
from cyberwave.client import _default_ca_bundle

DEBIAN_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"


def _paths(cafile):
    return ssl.DefaultVerifyPaths(
        cafile=cafile,
        capath=None,
        openssl_cafile_env="SSL_CERT_FILE",
        openssl_cafile="/etc/pki/tls/cert.pem",
        openssl_capath_env="SSL_CERT_DIR",
        openssl_capath="/etc/pki/tls/certs",
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: _paths(None))


def test_none_when_openssl_finds_a_default_cafile(monkeypatch):
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: _paths("/etc/ssl/cert.pem"))
    assert _default_ca_bundle() is None


def test_host_bundle_when_openssl_finds_no_cafile(monkeypatch):
    monkeypatch.setattr(client.os.path, "isfile", lambda path: path == DEBIAN_BUNDLE)
    assert _default_ca_bundle() == DEBIAN_BUNDLE


def test_certifi_when_no_cafile_and_no_host_bundle(monkeypatch):
    monkeypatch.setattr(client.os.path, "isfile", lambda path: False)
    assert _default_ca_bundle() == certifi.where()


def test_none_on_windows_even_without_cafile(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr(client.os.path, "isfile", lambda path: True)
    assert _default_ca_bundle() is None


def test_respects_ssl_cert_file(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", DEBIAN_BUNDLE)
    assert _default_ca_bundle() is None


def test_respects_ssl_cert_dir(monkeypatch):
    monkeypatch.setenv("SSL_CERT_DIR", "/etc/ssl/certs")
    assert _default_ca_bundle() is None
