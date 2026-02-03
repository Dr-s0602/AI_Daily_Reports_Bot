import os, json
from pathlib import Path
from datetime import datetime, timedelta, timezone

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

def get_drive_service():
    token_json = os.environ.get("GDRIVE_OAUTH_TOKEN_JSON")
    if not token_json:
        raise RuntimeError("GDRIVE_OAUTH_TOKEN_JSON이 없습니다. GitHub Secrets에 추가하세요.")
    creds = Credentials.from_authorized_user_info(json.loads(token_json), scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

def file_exists_in_folder(service, folder_id: str, file_name: str):
    q = f"'{folder_id}' in parents and name = '{file_name}' and trashed = false"
    res = service.files().list(q=q, fields="files(id,name)").execute()
    files = res.get("files", [])
    return (files[0]["id"] if files else None)

def upload_if_not_exists(service, folder_id: str, local_path: Path):
    file_name = local_path.name
    existed_id = file_exists_in_folder(service, folder_id, file_name)
    if existed_id:
        return "skipped", file_name

    media = MediaFileUpload(str(local_path), resumable=True)
    metadata = {"name": file_name, "parents": [folder_id]}
    service.files().create(body=metadata, media_body=media, fields="id").execute()
    return "created", file_name

def main():
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    if not folder_id:
        raise RuntimeError("GDRIVE_FOLDER_ID가 없습니다. GitHub Secrets에 추가하세요.")

    # KST 기준 “어제” 리포트를 업로드 (report 생성 로직과 맞춤)
    kst = timezone(timedelta(hours=9))
    target_date = (datetime.now(kst) - timedelta(days=1)).strftime("%Y-%m-%d")

    reports_dir = Path("reports")
    if not reports_dir.exists():
        print("⏭️ reports/ 폴더 없음 → 종료")
        return

    targets = [
        reports_dir / f"{target_date}_AI_Report.md",
        reports_dir / f"{target_date}_summaries.json",
    ]
    targets = [p for p in targets if p.exists()]

    if not targets:
        print(f"⏭️ 업로드할 파일 없음 ({target_date}) → 종료")
        return

    service = get_drive_service()

    print(f"📤 Drive 업로드 시작: {len(targets)}개 파일 ({target_date})")
    for p in targets:
        status, name = upload_if_not_exists(service, folder_id, p)
        print(f" - {status}: {name}")

    print("✅ Drive 미러링 완료")

if __name__ == "__main__":
    main()
