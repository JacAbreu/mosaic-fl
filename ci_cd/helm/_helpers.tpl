{{/*
Expand the name of the chart.
*/}}
{{- define "mosaicfl.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "mosaicfl.fullname" -}}
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
{{- define "mosaicfl.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mosaicfl.labels" -}}
helm.sh/chart: {{ include "mosaicfl.chart" . }}
{{ include "mosaicfl.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mosaicfl.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mosaicfl.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Server fullname
*/}}
{{- define "mosaicfl.server.fullname" -}}
{{ include "mosaicfl.fullname" . }}-server
{{- end }}

{{/*
Client fullname
*/}}
{{- define "mosaicfl.client.fullname" -}}
{{ include "mosaicfl.fullname" . }}-client-{{ .hospitalName }}
{{- end }}
