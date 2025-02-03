import requests
import os


def download_file(url, local_path, headers=None, verify_ssl=True):
    """
    Downloads a file from a URL to a specified local path.

    Parameters:
    - url (str): URL of the file to download
    - local_path (str): Full path where to save the file (including filename)
    - headers (dict): Optional custom headers to send with the request
    - verify_ssl (bool): Verify SSL certificates (default: True)

    Returns:
    - bool: True if download successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"File downloading to: {local_path}")
        response = requests.get(url, headers=headers, stream=True, verify=verify_ssl)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Write file in binary mode
        with open(local_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    file.write(chunk)

        print(f"File successfully downloaded to: {local_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {str(e)}")
    except IOError as e:
        print(f"File write error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

    return False

# Example usage:
# download_file(
#     url='https://example.com/image.jpg',
#     local_path='downloads/images/image.jpg',
#     headers={'User-Agent': 'Mozilla/5.0'},
#     verify_ssl=True
# )