import gdown

def download_from_gdrive():
    url = "https://drive.google.com/uc?id=1iQd89RCOuGryVoDzyWCnSulaj4kcm-I8"
    output = "asr_data.tar"
    gdown.download(url, output, quiet=False)

if __name__ == "__main__":
    download_from_gdrive()