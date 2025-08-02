# core/utils/network_patch.py
import ssl
import socket
import logging
import os

# --- Try to import necessary urllib3 components ---
try:
    import urllib3
    from urllib3.util import connection as urllib3_connection
    from urllib3.util import ssl_ as urllib3_ssl

    URLLIB3_AVAILABLE = True
except ImportError:
    URLLIB3_AVAILABLE = False
    logging.critical("urllib3 library not found. Network patch cannot be applied.")

# --- Try to import requests and socks for proxy patching ---
try:
    import requests
    import socks

    PROXY_LIBS_AVAILABLE = True
except ImportError:
    PROXY_LIBS_AVAILABLE = False
    logging.warning("requests or PySocks library not found. SOCKS5 proxy patch will not be applied.")


def apply_monkey_patch():
    """
    Applies several aggressive network patches to solve common connection issues.
    THIS IS A GLOBAL CHANGE AND POTENTIALLY INSECURE. USE WITH CAUTION.
    """
    logging.critical("--- ATTEMPTING TO APPLY AGGRESSIVE NETWORK PATCH ---")

    # --- Patch 1: Force IPv4 for urllib3 ---
    # This can solve connection timeout issues on networks with problematic IPv6.
    if URLLIB3_AVAILABLE:
        try:
            _original_allowed_gai_family = urllib3_connection.allowed_gai_family

            def force_ipv4():
                return socket.AF_INET

            urllib3_connection.allowed_gai_family = force_ipv4
            logging.critical("PATCH APPLIED: urllib3 is now forced to use IPv4.")
        except Exception as e:
            logging.error(f"Failed to apply IPv4 patch to urllib3: {e}")

    # --- Patch 2: Disable SSL Certificate Verification globally for urllib3 ---
    # This bypasses SSL errors but is a security risk as it allows man-in-the-middle attacks.
    if URLLIB3_AVAILABLE:
        try:
            _original_create_urllib3_context = urllib3_ssl.create_urllib3_context

            def create_unverified_context(**kwargs):
                # Creates a context that doesn't verify certificates
                kwargs['cert_reqs'] = ssl.CERT_NONE
                return _original_create_urllib3_context(**kwargs)

            urllib3_ssl.create_urllib3_context = create_unverified_context
            # Also disable the warning that comes with verify=False
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logging.critical(
                "PATCH APPLIED: SSL certificate verification for urllib3 is DISABLED globally. (Security Risk)")
        except Exception as e:
            logging.error(f"Failed to apply SSL verification patch to urllib3: {e}")

    # --- Patch 3: Force requests library to use SOCKS5 proxy from .env ---
    # This routes all `requests` traffic through the specified SOCKS5 proxy.
    if PROXY_LIBS_AVAILABLE:
        try:
            socks_host = os.getenv("SOCKS_PROXY_HOST")
            socks_port = os.getenv("SOCKS_PROXY_PORT")

            if socks_host and socks_port:
                socks_port = int(socks_port)

                # The core of the monkey patch
                _original_requests_getaddrinfo = socket.getaddrinfo

                def proxied_getaddrinfo(*args, **kwargs):
                    return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (args[0], args[1]))]

                # Set the default proxy for the entire socket module
                socks.set_default_proxy(socks.SOCKS5, socks_host, socks_port)

                # Replace the socket's getaddrinfo with our proxied version
                socket.getaddrinfo = proxied_getaddrinfo

                logging.critical(
                    f"PATCH APPLIED: All `requests` traffic is now routed through SOCKS5 proxy at {socks_host}:{socks_port}.")
            else:
                logging.info("SOCKS_PROXY_HOST or SOCKS_PROXY_PORT not found in .env. Skipping SOCKS5 proxy patch.")
        except Exception as e:
            logging.error(f"Failed to apply SOCKS5 proxy patch to requests: {e}")

    logging.critical("--- NETWORK PATCH ATTEMPT COMPLETE ---")