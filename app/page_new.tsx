'use client'
import { useState, useRef, useEffect } from 'react'
import { Upload, FileText, Target, Download, Brain, BarChart3, Zap, TrendingUp, Shield, CheckCircle, AlertCircle, Activity, Network, Database, Play, Briefcase, Code, Globe, Sparkles } from 'lucide-react'

type SystemStatus = { status: string; ml_engine: boolean; groq_available: boolean } | null
type Analysis = { atsScore: number; atsFeedback: string; techFeedback: string; hrFeedback: string; skillGaps: string[]; recommendations: string[] } | null
type GraphNode = { id: string; type: string; weight: number }
type GraphEdge = { source: string; target: string; weight: number }
type GraphData = { resume_graph: { nodes: GraphNode[]; edges: GraphEdge[] }; jd_graph: { nodes: GraphNode[]; edges: GraphEdge[] } } | null
type DetailedResults = { gat_matching?: { graph_similarity: number; skill_overlap: string[]; missing_skills: string[]; match_score: number }; bias_detection?: { bias_detected: boolean; bias_types: string[]; quality_score: number }; career_trajectory?: { predicted_roles: string[]; skill_development: string[]; timeline: string } } | null