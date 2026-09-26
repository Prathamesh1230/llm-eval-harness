"""SSRF protection for user-supplied target URLs.

On a public deployment, visitors choose which URL our server calls. Without
checks, they could point it at internal addresses: localhost, the private
network, or the cloud metadata service at 169.254.169.254, which can leak
the server's own credentials. This module resolves the hostname and rejects
anything that isn't a public internet address.

Known limitation: DNS rebinding. A hostile DNS server can return a public IP
when we check and a private one when we connect. The HTTP adapter reduces
this by refusing redirects; a full fix pins the checked IP for the request.
"""

import ipaddress
import os
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = {"http", "https"}


class UnsafeTargetError(ValueError):
    """Raised when a URL is not safe for the server to call."""


def private_targets_allowed() -> bool:
    """Self-hosted installs may test internal chatbots; public ones must not.

    Set ALLOW_PRIVATE_TARGETS=true in .env for local development or when a
    company runs the harness inside its own network. Leave it off on any
    public deployment.
    """
    return os.getenv("ALLOW_PRIVATE_TARGETS", "false").lower() == "true"


def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # "::ffff:127.0.0.1" is localhost wearing an IPv6 disguise — unwrap it.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def validate_target_url(url: str, allow_private: bool | None = None) -> str:
    """Return the URL if it is safe to call, otherwise raise UnsafeTargetError."""
    if allow_private is None:
        allow_private = private_targets_allowed()

    parsed = urlparse(url.strip())

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeTargetError(f"Only http and https are allowed, got '{parsed.scheme or 'none'}'")
    if not parsed.hostname:
        raise UnsafeTargetError("URL has no hostname")
    if parsed.username or parsed.password:
        raise UnsafeTargetError("Credentials inside the URL are not allowed; use headers instead")

    if allow_private:
        return url

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise UnsafeTargetError(f"Could not resolve host '{parsed.hostname}'")

    # A hostname can resolve to several IPs. Every one must be public,
    # otherwise an attacker could mix one safe IP with one internal IP.
    for info in infos:
        raw_ip = info[4][0].split("%")[0]  # drop IPv6 zone ids like "%eth0"
        ip = ipaddress.ip_address(raw_ip)
        if not _is_public(ip):
            raise UnsafeTargetError(
                f"'{parsed.hostname}' resolves to a non-public address ({ip}); blocked"
            )

    return url


if __name__ == "__main__":
    cases = [
        ("http://localhost:8000/chat", False),
        ("http://127.0.0.1/chat", False),
        ("http://169.254.169.254/latest/meta-data", False),
        ("http://10.0.0.5/chat", False),
        ("http://192.168.1.1/chat", False),
        ("ftp://example.com/file", False),
        ("http://user:pass@example.com/chat", False),
        ("https://example.com/chat", True),
    ]

    print("Public mode (ALLOW_PRIVATE_TARGETS off):")
    for url, should_pass in cases:
        try:
            validate_target_url(url, allow_private=False)
            result = "allowed"
        except UnsafeTargetError as e:
            result = f"blocked  ({e})"
        ok = "OK  " if (result == "allowed") == should_pass else "FAIL"
        print(f"  {ok} {url:<45} {result}")

    print("\nSelf-hosted mode (ALLOW_PRIVATE_TARGETS on):")
    validate_target_url("http://localhost:8000/chat", allow_private=True)
    print("  OK   http://localhost:8000/chat                    allowed")