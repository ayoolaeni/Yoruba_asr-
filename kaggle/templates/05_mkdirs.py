# ---- Package source: create folders, then the %%writefile cells below write every file -------
for _d in {{DIRS}}:
    os.makedirs(f"{CODE_DIR}/{_d}", exist_ok=True)
print("writing the yoruba_asr package to", CODE_DIR)
