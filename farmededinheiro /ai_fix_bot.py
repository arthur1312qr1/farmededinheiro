import os
import argparse
import google.generativeai as genai
from datetime import datetime

# Configure a API da Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_corrected_code(error_log, file_path):
    """
    Usa a IA da Gemini para analisar um log de erro e corrigir um arquivo.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    prompt = f"""
    Eu sou um assistente de IA. Um bot de negociação em Python falhou e eu preciso de ajuda para corrigi-lo.

    Aqui está o log de erro completo:
    ---
    {error_log}
    ---

    Aqui está o conteúdo completo do arquivo {file_path}:
    ---
    {file_content}
    ---

    A minha tarefa é analisar o erro, entender o contexto do código e fornecer o código completo e corrigido para o arquivo {file_path}.
    É crucial que você não responda com explicações, mas apenas com o código completo e corrigido.

    O código corrigido deve ser o arquivo {file_path} inteiro, pronto para ser salvo.
    """

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

def main():
    parser = argparse.ArgumentParser(description="Bot de correção de erros com IA")
    parser.add_argument("error_log_path", help="Caminho para o arquivo de log de erro")
    parser.add_argument("file_to_fix_path", help="Caminho para o arquivo que precisa ser corrigido")
    args = parser.parse_args()

    # Lê o log de erro
    with open(args.error_log_path, 'r', encoding='utf-8') as f:
        error_log = f.read()
    
    file_to_fix = args.file_to_fix_path
    print(f"Analisando o log e o arquivo {file_to_fix} com a Gemini...")

    # Obtém o código corrigido da IA
    corrected_code = get_corrected_code(error_log, file_to_fix)
    
    # Remove qualquer formatação extra que a IA possa ter adicionado
    if corrected_code.startswith("```python") and corrected_code.endswith("```"):
        corrected_code = corrected_code[len("```python"): -len("```")].strip()

    # Salva o código corrigido no arquivo
    with open(file_to_fix, 'w', encoding='utf-8') as f:
        f.write(corrected_code)
    
    print(f"Arquivo {file_to_fix} corrigido e salvo.")
    
if __name__ == "__main__":
    main()
