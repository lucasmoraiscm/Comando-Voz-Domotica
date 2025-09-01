from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from pydub import AudioSegment
import requests
from requests.exceptions import HTTPError
import json
import os
import io
import re


# Carrega o arquivo .env
load_dotenv()


# Cria app Flask
app = Flask(__name__)
CORS(app)


# Define o Web Socket
socketio = SocketIO(app, cors_allowed_origins="*")


def configurar_gemini():
    try:
        # Busca a chave da API do Gemini armazenada no arquivo .env
        api_key = os.getenv("GEMINI_API_KEY")

        # Verifica se a chave da API do Gemini foi encontrada
        if not api_key:
            raise ValueError("A chave da API do Gemini não foi encontrada!")
        
        # Configura a chave da API do Gemini e define o modelo
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Retorna o modelo configurado
        return model
    
    except Exception as e:
        print(f"Erro ao configurar o Gemini: {e}")
        return None


def enviar_prompt_audio(audio):
    try:
        print(f"Recebido arquivo: {audio.filename}")

        # Configura o modelo do Gemini
        model = configurar_gemini()

        # Receber lista de itens
        itens = listar_itens()

        if itens is None:
            return "Erro: Não foi possível obter a lista de itens"
        
        # Converte o dicionário para uma string JSON e depois para bytes
        itens_bytes = json.dumps(itens).encode('utf-8')

        # Cria um arquivo JSON em memória
        json_buffer = io.BytesIO(itens_bytes)
        json_buffer.seek(0)

        print("Fazendo upload do arquivo JSON de itens para o Gemini...")
        # Realiza o upload do arquivo JSON para o Gemini
        itens_file = genai.upload_file(
            path=json_buffer,
            display_name="lista_itens",
            mime_type="text/plain"
        )
        print("Upload do JSON concluído!")

        # Carrega o áudio para o processamento
        audio_completo = AudioSegment.from_file(audio)

        # Cria um áudio em memória
        audio_buffer = io.BytesIO()
        audio_completo.export(audio_buffer, format="wav")
        audio_buffer.seek(0)

        print("Fazendo upload do áudio para o Gemini...")
        # Configura o áudio para ser enviado ao Gemini
        audio_file = genai.upload_file(
            path=audio_buffer,
            display_name="audio_completo",
            mime_type="audio/wav"
        )
        print("Upload do áudio concluído!")

        # Define um histórico com contexto para o Gemini
        historico_inicial = [
            {"role": "user", "parts": [
                {"text": """
Você é um assistente inteligente residencial que deve analisar comandos de voz procurando por itens presentes em uma lista de itens. 
Vou te passar um arquivo JSON com a lista de itens e um comando de voz.
O arquivo JSON contém conjuntos de itens onde cada item apresenta sua entidade e o seu nome.
Analise o comando de voz e o arquivo JSON para responder.
                
Regras de resposta:
- Se existir um item na lista de itens do conjunto "dispositivos" com o mesmo nome de algum item citado no comando de voz então retorne "{"entidade": [entidade do item], "nome": [nome do item], "acao": [(ligar ou desligar)]}" sem formatação JSON.
- Se existir um item na lista de itens do conjunto "grupos" com o mesmo nome de algum item citado no comando de voz então retorne "{"entidade": [entidade do item], "nome": [nome do item], "acao": [(ligar ou desligar)]}" sem formatação JSON.
- Se existir um item na lista de itens do conjunto "cenas" com o mesmo nome de algum item citado no comando de voz então retorne "{"entidade": [entidade do item], "nome": [nome do item], "acao": [(ligar ou desligar)]}" sem formatação JSON.
- Se existir um item na lista de itens do conjunto "acoesCena" com o mesmo nome de algum item citado no comando de voz e for solicitado que ele seja executado então retorne "{"entidade": [entidade do item], "nome": [nome do item], "acao": "executar"}" sem formatação JSON.
- Se não existir um item na lista de itens com o mesmo nome de algum item citado no comando de voz então retorne "{"entidade": null, "nome": null, "acao": null}" sem formatação JSON.
- Se receber um comando de voz que não tenha relação com itens de uma residência então retorne "{"entidade": null, "nome": null, "acao": null}" sem formatação JSON.
                
Sua resposta deve conter ÚNICA E EXCLUSIVAMENTE o objeto JSON solicitado, sem nenhum texto, explicação ou formatação markdown adicional.

Exemplos de respostas válidas:
{"entidade": "Dispositivo", "nome": "Luz Sala", "acao": "ligar"}
{"entidade": "AcaoCena", "nome": "Cinema", "acao": "executar"}
{"entidade": null, "nome": null, "acao": null}
Não inclua textos como "Claro, aqui está:" ou qualquer outra coisa fora do objeto JSON.
                """}
            ]},
            {"role": "model", "parts": [{"text": "Entendido. Por favor, envie o arquivo de itens e o comando de voz."}]}
        ]

        # Configura o modelo Gemini com o histórico que foi definido
        chat_session = model.start_chat(history=historico_inicial)

        # Define o prompt com o arquivo de áudio configurado
        prompt = [
            "Analise o comando de voz a seguir usando a lista de itens fornecida no arquivo JSON.",
            itens_file,
            audio_file
        ]
        print("Prompt enviado!")
        
        # Realiza a requisição e recebe a resposta do Gemini
        response = chat_session.send_message(prompt)
        print("Resposta recebida!")

        # Limpa o arquivo de áudio enviado ao Gemini que já foi processado
        genai.delete_file(itens_file.name)
        genai.delete_file(audio_file.name)
        print(f"Arquivo JSON e áudio limpos.")

        print("Processando resposta do Gemini...")
        # Envia a resposta do Gemini para processar a ação solicitada
        resposta_processada = processar_resposta_gemini(response.text)

        # Retorna o texto do processamento da ação solicitada
        return resposta_processada

    except Exception as e:
        print(f"Ocorreu um erro na API do Gemini: {e}")
        return f"Erro ao processar com a API do Gemini: {e}"


def listar_itens():
    url = "http://31.97.22.121:8080/history"

    try:
        # Envia a requisição para a url
        response = requests.get(url, timeout=10)
        
        # Verifica se a requisição foi bem-sucedida
        response.raise_for_status()
        
        # Converte a resposta JSON em um dicionário Python
        itens = response.json()
        
        return itens
    
    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar dispositivos da API: {e}")
        return None


def processar_resposta_gemini(resposta_gemini):
    try:
        # Utiliza a expressão regular para encontrar o JSON
        match = re.search(r'\{.*\}', resposta_gemini, re.DOTALL)

        # Se não encontrar o JSON
        if not match:
            print(f"Nenhum JSON encontrado na resposta: {resposta_gemini}")
            return "Não foi possível identificar a solicitação na resposta do assistente."
        
        # Recebe o JSON e carrega sua estrutura para manipulação
        json_string = match.group(0)
        resposta_gemini_json = json.loads(json_string)

        # Recebe as informações do JSON
        entidade = resposta_gemini_json.get("entidade")
        nome = resposta_gemini_json.get("nome")
        acao = resposta_gemini_json.get("acao")

        # Se o JSON tiver informações nulas
        if entidade is None and nome is None and acao is None:
            return "Não foi possível realizar a solicitação. Tente novamente"
        
        # Converte a ação para caixa baixa para evitar problemas em futuras requisições
        acao = acao.lower()

        # Busca o id do item especificado
        id_item = buscar_id(entidade, nome)

        # Se não encontrar um id para o item especificado
        if not id_item:
            return f"O item com nome '{nome}' não foi encontrado."

        # Executa a ação solicitada para o item especifico
        response_acao = executar_acao(entidade, acao, id_item)

        # Retorna a resposta da execução da ação
        return response_acao

    except json.JSONDecodeError:
        print(f"Erro ao decodificar JSON da resposta: {resposta_gemini}")
        return "A resposta do assistente não estava em um formato válido."
    except requests.exceptions.RequestException as e:
        print(f"Erro ao conectar com a API de dispositivos: {e}")
        return "Falha ao buscar dispositivos."
    except Exception as e:
        print(f"Erro ao processar resposta: {e}")
        return "Erro interno no processamento."
    

def buscar_id(entidade, nome):
    # Define a url e a nomenclatura do atributo id de acordo com a entidade
    match entidade:
        case "Dispositivo":
            url = "http://31.97.22.121:8080/dispositivos"
            id_entidade = "idDispositivo"
            
        case "Cena":
            url = "http://31.97.22.121:8080/cenas"
            id_entidade = "idCena"

        case "AcaoCena":
            url = "http://31.97.22.121:8080/acaocenas"
            id_entidade = "idAcao"
            
        case "Grupo":
            url = "http://31.97.22.121:8080/grupos"
            id_entidade = "idGrupo"

        case _:
            return "Entidade não encontrada"

    # Envia a requisição para a url
    response = requests.get(url, timeout=10)

    # Verifica se a requisição foi bem-sucedida
    response.raise_for_status()

    # Converte a resposta JSON em um dicionário Python
    itens = response.json()

    # Verifica se existe um item com o mesmo nome que foi especificado e retorna seu id
    for item in itens:
        if item["nome"] == nome:
            return item[id_entidade]
    
    # Se não for encontrado um item com o mesmo nome que foi especificado não retorna um valor
    return None


def executar_acao(entidade, acao, id):
    try:
        # Valida a ação solicitada e define a url de acordo com a entidade
        match entidade:
            case "Dispositivo":
                if (acao != "ligar") and (acao != "desligar"):
                    return "Ação não identificada para dispositivos"

                url = f"http://31.97.22.121:8080/dispositivos/{id}/{acao}"
                
            case "Cena":
                if (acao != "ligar") and (acao != "desligar"):
                    return "Ação não identificada para cenas"
                
                url = f"http://31.97.22.121:8080/cenas/{id}/{acao}"

            case "AcaoCena":
                if acao != "executar":
                    return "Ação não identificada para ações de cenas"
                
                url = f"http://31.97.22.121:8080/acaocenas/{id}/{acao}"
                
            # A entidade Grupo possui execução própria pois sua requisição é com método POST
            case "Grupo":
                if (acao != "ligar") and (acao != "desligar"):
                    return "Ação não identificada para grupos"
                
                url = f"http://31.97.22.121:8080/grupos/{id}/{acao}"

                # Envia a requisição para a url
                response_grupo = requests.post(url, timeout=10)

                # Verifica se a requisição foi bem-sucedida
                response_grupo.raise_for_status()

                # Retorna o texto da resposta da ação executada
                return response_grupo.text

            case _:
                return "Entidade não encontrada"
        
        # Envia a requisição para a url
        response = requests.put(url, timeout=10)

        # Verifica se a requisição foi bem-sucedida
        response.raise_for_status()

        # Retorna o texto da resposta da ação executada
        return response.text

    except HTTPError as http_err:
        if http_err.response.status_code == 400:
            return http_err.response.text
        else:
            return f"Erro HTTP inesperado: {http_err.response.status_code} - {http_err.response.text}"

    except requests.exceptions.RequestException as e:
        print(f"Erro ao conectar com a API de ação: {e}")
        return "Falha ao executar ação."


@app.route('/processar-audio', methods=['POST'])
def processar_audio():
    # Verifica se o arquivo de áudio foi enviado na requisição
    if 'audio_file' not in request.files:
        return jsonify({"error": "Nenhum arquivo de áudio foi enviado"}), 400
    
    # Armazena o arquivo de áudio recebido da requisição
    file = request.files['audio_file']

    # Verifica se o arquivo de áudio foi selecionado
    if file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400
    
    # Envia o arquivo de áudio para o prompt e armazena a resposta
    response = enviar_prompt_audio(file)

    # Verifica se houve um erro durante a configuração do prompt
    if "Erro interno" in response:
        return jsonify({"error": response}), 500
    
    # Retorna a resposta do prompt enviado
    return jsonify({"text": response})


if __name__ == "__main__":
    # Executa o app Flask com o Web Socket configurado
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
