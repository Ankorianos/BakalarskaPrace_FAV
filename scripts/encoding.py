import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result
    
def open_file_with_encoding(file_path, encoding):
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()

if __name__ == "__main__":
    #chci pro kazdej soubor ve slozce directory, zjistit encoding a otevrit ho s tim encodingem, a vypis prvnich 500 znaku pro kontrolu
    import os
    directory = "../INTERSPEECH2023"
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            encoding_info = detect_encoding(file_path)
            #opened_text = open_file_with_encoding(file_path, encoding_info['encoding'])
            #print(opened_text[:500])  # vypíše prvních 500 znaků pro kontrolu
            print(file_path)           
            print("Encoding:", encoding_info['encoding'])
            print("Confidence:", encoding_info['confidence']) 

            #trs a formal for evaluation cp1250