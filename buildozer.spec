[app]
title = Face Converter
package.name = faceconverter
package.domain = org.efootballtools
source.dir = .
source.include_exts = py,json,glb,npz,txt,md,png
source.exclude_dirs = .github,.venv,input,output,_work,__pycache__,bin
version = 1.0
requirements = python3,kivy==2.3.0,numpy,pillow
orientation = portrait
fullscreen = 0

android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,MANAGE_EXTERNAL_STORAGE
android.api = 33
android.minapi = 24
android.ndk = 25b
android.archs = arm64-v8a
android.accept_sdk_license = True
android.allow_backup = True

[buildozer]
log_level = 2
warn_on_root = 1
