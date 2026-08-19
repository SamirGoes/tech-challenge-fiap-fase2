# Imagem de container para rodar a API como AWS Lambda.
# Preserva a mesma estrutura de pastas do repositório (src/ ao lado de api/)
# porque api/main.py resolve os caminhos de forma relativa a essa estrutura.
FROM public.ecr.aws/lambda/python:3.12

WORKDIR ${LAMBDA_TASK_ROOT}

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY src/ src/
COPY api/ api/

CMD ["api.main.handler"]
