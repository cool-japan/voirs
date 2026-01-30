{{/*
Expand the name of the chart.
*/}}
{{- define "voirs-evaluation.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "voirs-evaluation.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "voirs-evaluation.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "voirs-evaluation.labels" -}}
helm.sh/chart: {{ include "voirs-evaluation.chart" . }}
{{ include "voirs-evaluation.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "voirs-evaluation.selectorLabels" -}}
app.kubernetes.io/name: {{ include "voirs-evaluation.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "voirs-evaluation.serviceAccountName" -}}
{{- if .Values.rbac.serviceAccount.create }}
{{- default (include "voirs-evaluation.fullname" .) .Values.rbac.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.rbac.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Image name
*/}}
{{- define "voirs-evaluation.image" -}}
{{- $registry := .Values.image.registry }}
{{- if .Values.global.imageRegistry }}
{{- $registry = .Values.global.imageRegistry }}
{{- end }}
{{- printf "%s/%s:%s" $registry .Values.image.repository .Values.image.tag }}
{{- end }}

{{/*
Return the proper Docker Image Pull Secrets
*/}}
{{- define "voirs-evaluation.imagePullSecrets" -}}
{{- if or .Values.image.pullSecrets .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.global.imagePullSecrets }}
  - name: {{ . }}
{{- end }}
{{- range .Values.image.pullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Return the proper Storage Class
*/}}
{{- define "voirs-evaluation.storageClass" -}}
{{- if .Values.global.storageClass }}
{{- .Values.global.storageClass }}
{{- else if .Values.coordinator.persistence.storageClass }}
{{- .Values.coordinator.persistence.storageClass }}
{{- end }}
{{- end }}
