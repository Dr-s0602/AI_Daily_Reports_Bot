import os
import json
from pathlib import Path
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def get_drive_service():
    sa_json = os.environ.get("GCP_SA_KEY_JSON")
    if not sa_json:
        raise RuntimeError("GCP_SA_KEY_JSON 환경변수가 없습니다. GitHub Secrets에 추가하세요.")

    sa_info = json.loads(sa_json)
    creds = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

def upload_if_not_exists(service, folder_id: str, local_path: Path):
    """
    같은 파일명이 Drive 폴더에 이미 있으면 업로드 스킵.
    없을 때만 create.
    """
    file_name = local_path.name

    # 폴더 내 동일 파일명 검색
    q = (
        f"'{folder_id}' in parents and "
        f"name = '{file_name}' and "
        f"trashed = false"
    )
    res = service.files().list(q=q, fields="files(id, name)").execute()
    files = res.get("files", [])

    if files:
        return "skipped", file_name

    media = MediaFileUpload(str(local_path), resumable=True)
    metadata = {"name": file_name, "parents": [folder_id]}
    service.files().create(body=metadata, media_body=media, fields="id").execute()
    return "created", file_name


def upload_or_update_file(service, folder_id: str, local_path: Path):
    """
    같은 파일명이 Drive 폴더에 있으면 update, 없으면 create.
    """
    file_name = local_path.name

    # 폴더 내 동일 파일명 검색
    q = (
        f"'{folder_id}' in parents and "
        f"name = '{file_name}' and "
        f"trashed = false"
    )
    res = service.files().list(q=q, fields="files(id, name)").execute()
    files = res.get("files", [])

    media = MediaFileUpload(str(local_path), resumable=True)

    if files:
        file_id = files[0]["id"]
        service.files().update(fileId=file_id, media_body=media).execute()
        return "updated", file_name
    else:
        metadata = {"name": file_name, "parents": [folder_id]}
        service.files().create(body=metadata, media_body=media, fields="id").execute()
        return "created", file_name


def main():
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    if not folder_id:
        raise RuntimeError("GDRIVE_FOLDER_ID 환경변수가 없습니다. GitHub Secrets에 추가하세요.")

    reports_dir = Path("reports")
    if not reports_dir.exists():
        raise RuntimeError("reports/ 폴더가 없습니다. 리포트 생성 단계가 먼저 실행되어야 합니다.")

    # 업로드 대상: reports 폴더의 md/json 전부 (필요하면 필터링 가능)
    targets = list(reports_dir.glob("*.md")) + list(reports_dir.glob("*.json"))
    if not targets:
        raise RuntimeError("업로드할 파일이 없습니다. reports/*.md, *.json을 확인하세요.")

    service = get_drive_service()

    print(f"📤 Drive 업로드 시작: {len(targets)}개 파일")
    for p in targets:
        status, name = upload_if_not_exists(service, folder_id, p)
        print(f" - {status}: {name}")

    print("✅ Drive 미러링 완료")


if __name__ == "__main__":
    main()
