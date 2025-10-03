# 1. Usar uma imagem base leve do Python
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# 2. Definir diretório de trabalho
WORKDIR /app

# 3. Copiar requirements (se tiver dependências)
COPY requirements.txt .

# 4. Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar os arquivos da aplicação
COPY . .

# 6. Expor a porta 5000 (padrão Flask)
EXPOSE 5000

# 7. Rodar a aplicação
CMD ["python", "app.py"]
