"""
KONOMI Protocol - Server Main
Entry point for running the server
"""

import asyncio


async def main():
    """Run server."""
    from core.konomi import Konomi
    from .core import DialUpServer

    konomi = Konomi()
    server = DialUpServer(konomi)
    await server.start()


if __name__ == '__main__':
    asyncio.run(main())
