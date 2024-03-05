import os
import subprocess
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']
creds = None
if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file('scripts\credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

service = build('drive', 'v3', credentials=creds)

def upload_file(file_name, parent_id=None):
    file_metadata = {
        'name': file_name.split('/')[-1],
        'parents': [parent_id] if parent_id else []
    }
    media = MediaFileUpload(file_name, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"Uploaded {file_name} to Google Drive with ID: {file.get('id')}")

def run_and_upload(src_folder, output_folder_id):
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith('.py') or file.endswith('.ipynb'):
                # Determine output path
                output_file = os.path.join('outputs', os.path.relpath(root, src_folder), file + '.output')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Run the file
                if file.endswith('.py'):
                    subprocess.run(['python', os.path.join(root, file)], stdout=open(output_file, 'w'))
                else:  # For .ipynb files
                    subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
                                    '--output', output_file, os.path.join(root, file)])
                
                # Upload the output
                upload_file(output_file, parent_id=output_folder_id)


src_folder = '../src'
output_folder_id = 'https://drive.google.com/drive/folders/1mpUaQ5qhB0aRNIQysZHjdsuFQLzm0QXz?usp=sharing'  
run_and_upload(src_folder, output_folder_id)
