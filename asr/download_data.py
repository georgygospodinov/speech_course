import gdown

def download_from_gdrive():
    url = "https://drive.google.com/uc?id=1TEOR60JXgOkPrC6jSLhuR2Nb6eCegjpd"
    output = "asr_data.tar"
    gdown.download(url, output, quiet=False)

if __name__ == "__main__":
    download_from_gdrive()