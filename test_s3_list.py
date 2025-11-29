#!/usr/bin/env python3
"""
Test S3 Flatfiles Access
=========================
실제 파일 목록 조회하여 경로 확인
"""

import boto3
from botocore.config import Config
import os

# Credentials
ACCESS_KEY_ID = 'f0bc904a-9d5c-476b-af56-2cb4a2455a3e'
SECRET_ACCESS_KEY = 'w7KprL4_lK7uutSH0dYGARkucXHOFXCN'
ENDPOINT_URL = 'https://files.massive.com'
BUCKET = 'flatfiles'

print("=" * 80)
print("Testing Massive.com Flatfiles Access")
print("=" * 80)
print()

# S3 클라이언트 생성
print("1. Creating S3 client...")
s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY,
    endpoint_url=ENDPOINT_URL,
    config=Config(signature_version='s3v4')
)
print("✅ S3 client created")
print()

# 파일 목록 조회
print("2. Listing objects in bucket 'flatfiles'...")
print()

try:
    # us_stocks_sip 프리픽스로 조회
    paginator = s3_client.get_paginator('list_objects_v2')
    
    print("Prefix: us_stocks_sip/")
    print("-" * 80)
    
    count = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix='us_stocks_sip/', MaxKeys=100):
        if 'Contents' in page:
            for obj in page['Contents']:
                print(f"  {obj['Key']}")
                count += 1
                
                if count >= 50:  # 처음 50개만 출력
                    print(f"\n  ... (showing first 50 objects)")
                    break
        
        if count >= 50:
            break
    
    print()
    print(f"✅ Found {count}+ objects")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    
    # 루트 디렉토리 조회 시도
    print("Trying to list root directory...")
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET, Delimiter='/', MaxKeys=100)
        
        if 'CommonPrefixes' in response:
            print("\nAvailable prefixes:")
            for prefix in response['CommonPrefixes']:
                print(f"  {prefix['Prefix']}")
        
        if 'Contents' in response:
            print("\nFiles in root:")
            for obj in response['Contents']:
                print(f"  {obj['Key']}")
                
    except Exception as e2:
        print(f"❌ Error listing root: {e2}")

print()
print("=" * 80)
