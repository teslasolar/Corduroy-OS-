"""
Konomi Corduroy-OS - Version Shim
Spoofs Windows version to report Windows 10

This module provides utilities for version spoofing.
In a real implementation, this would be a DLL injected
into XP applications to make them think they're running
on Windows 10.
"""

# Windows 10 version info (Build 19041 = 2004 / 20H1)
VERSION_INFO = {
    'major': 10,
    'minor': 0,
    'build': 19041,
    'platform': 'NT',
    'service_pack': '',
    'product_name': 'Windows 10 Pro'
}


def get_version_ex():
    """
    Simulate GetVersionEx API call.

    Returns Windows 10 version info.

    Returns:
        dict: Version information
    """
    return VERSION_INFO.copy()


def get_version_string():
    """
    Get version as string.

    Returns:
        str: "10.0.19041"
    """
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['build']}"


def spoof_api_call(api_name: str, *args, **kwargs):
    """
    Spoof Windows API calls for XP app compatibility.

    Args:
        api_name: Name of API being called
        *args: API arguments
        **kwargs: API keyword arguments

    Returns:
        Spoofed return value
    """
    if api_name == 'GetVersionExA' or api_name == 'GetVersionExW':
        return get_version_ex()

    # Add more API spoofs as needed
    return None


# C code template for version.dll shim
VERSION_DLL_TEMPLATE = """
// version.dll - Windows Version Spoofing Shim
// Compile: gcc -shared -o version.dll version.c

#include <windows.h>

typedef struct _OSVERSIONINFOA {
    DWORD dwOSVersionInfoSize;
    DWORD dwMajorVersion;
    DWORD dwMinorVersion;
    DWORD dwBuildNumber;
    DWORD dwPlatformId;
    CHAR  szCSDVersion[128];
} OSVERSIONINFOA;

BOOL WINAPI GetVersionExA_Shim(OSVERSIONINFOA *lpVersionInfo) {
    if (!lpVersionInfo || lpVersionInfo->dwOSVersionInfoSize < sizeof(OSVERSIONINFOA)) {
        return FALSE;
    }

    // Return Windows 10 version (Build 19041)
    lpVersionInfo->dwMajorVersion = 10;
    lpVersionInfo->dwMinorVersion = 0;
    lpVersionInfo->dwBuildNumber = 19041;
    lpVersionInfo->dwPlatformId = VER_PLATFORM_WIN32_NT;
    strcpy(lpVersionInfo->szCSDVersion, "");

    return TRUE;
}

// Export as GetVersionExA
#pragma comment(linker, "/export:GetVersionExA=GetVersionExA_Shim")

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    return TRUE;
}
"""


if __name__ == '__main__':
    print("Konomi Corduroy-OS - Version Shim")
    print(f"Spoofed Version: {get_version_string()}")
    print(f"Full Info: {get_version_ex()}")
    print("\nTo create version.dll:")
    print("1. Save VERSION_DLL_TEMPLATE to version.c")
    print("2. Compile: gcc -shared -o version.dll version.c")
    print("3. Place version.dll in system32")
