import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';

export interface PredictResponse {
  predicao: string;
  resultado: string;
  tem_doenca: boolean;
  chance_doenca: number;
  probabilidade: number;
  explicacao: string | null;
}

export interface ResultadoLoteItem {
  linha: number;
  id: string | null;
  predicao: string;
  resultado: string;
  tem_doenca: boolean;
  chance_doenca: number;
  probabilidade: number;
  diagnostico_real: string | null;
  acertou: boolean | null;
  explicacao: string | null;
}

export interface PredictLoteResponse {
  total: number;
  positivos: number;
  negativos: number;
  acertos: number | null;
  resultados: ResultadoLoteItem[];
  erros: { linha?: number; erro?: string }[];
}

export interface FeatureSimplificado {
  chave: string;
  rotulo: string;
  importancia: number;
}

export interface FeaturesSimplificadoResponse {
  algoritmo: string;
  metodo: string;
  n: number;
  metricas_holdout: Record<string, number>;
  features: FeatureSimplificado[];
}

@Injectable({ providedIn: 'root' })
export class Api {
  private readonly http = inject(HttpClient);

  prever(features: Record<string, number>) {
    return this.http.post<PredictResponse>('/predict', { features });
  }

  preverSimplificado(features: Record<string, number>) {
    return this.http.post<PredictResponse>('/predict/simplificado', { features });
  }

  preverLote(arquivo: File) {
    const dados = new FormData();
    dados.append('arquivo', arquivo, arquivo.name);
    return this.http.post<PredictLoteResponse>('/predict/lote', dados);
  }

  preverLoteSimplificado(arquivo: File) {
    const dados = new FormData();
    dados.append('arquivo', arquivo, arquivo.name);
    return this.http.post<PredictLoteResponse>('/predict/lote/simplificado', dados);
  }

  listarSimplificado() {
    return this.http.get<FeaturesSimplificadoResponse>('/predict/simplificado/features');
  }
}
