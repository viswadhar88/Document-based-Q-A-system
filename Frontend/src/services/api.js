import axios from 'axios';
import toast from 'react-hot-toast';
import websocketService from './websocket';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}/api/v1` : '/api/v1',
  timeout: 90000, 
  headers: {
    'Content-Type': 'application/json', // Clean JSON headers
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    config.metadata = { startTime: new Date() };
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    const endTime = new Date();
    const duration = endTime - response.config.metadata.startTime;
    response.duration = duration;
    return response;
  },
  (error) => {
    if (error.response) {
      const { status, data } = error.response;
      
      switch (status) {
        case 400:
          toast.error(data.detail || 'Invalid request');
          break;
        case 401:
          toast.error('Unauthorized access');
          break;
        case 403:
          toast.error('Access forbidden');
          break;
        case 404:
          toast.error('Resource not found');
          break;
        case 413:
          toast.error('File too large');
          break;
        case 422:
          console.error("Validation Error Details:", data);
          toast.error('Validation Error: Check your request data format');
          break;
        case 500:
          toast.error('Server error. Please try again.');
          break;
        default:
          toast.error('Something went wrong');
      }
    } else if (error.request) {
      toast.error('Network error. Please check your connection.');
    } else {
      toast.error('Request failed');
    }
    
    return Promise.reject(error);
  }
);

// Document API functions
export const documentAPI = {
  uploadDocuments: async (files, onProgress) => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file); 
    });
    
    formData.append('client_id', websocketService.getClientId());

    const response = await api.post('/documents/upload-with-progress', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        onProgress?.(percentCompleted);
      },
    });

    return response.data;
  },

  getDocuments: async () => {
    const response = await api.get('/documents');
    return response.data;
  },

  deleteDocument: async (documentId) => {
    const response = await api.delete(`/documents/${documentId}`);
    return response.data;
  },

  getDocumentContent: async (documentId) => {
    const response = await api.get(`/documents/${documentId}/content`);
    return response.data;
  },

  // 🚀 Unified JSON Search
  searchDocuments: async (query, options = {}) => {
    const response = await api.post('/documents/search', {
      query: query,
      top_k: options.limit || options.top_k || 5,
      score_threshold: options.threshold || options.score_threshold || 0.3,
      // Pass the array directly! No more joining with commas.
      document_ids: options.document_ids && options.document_ids.length > 0 ? options.document_ids : null
    });
    
    return response.data;
  },
};

// Query API functions
export const queryAPI = {
  askQuestion: async (question, documentIds = [], options = {}) => {
    const response = await api.post('/query/ask', {
      question,
      document_ids: documentIds,
      top_k: options.top_k || 5,
      score_threshold: options.score_threshold || 0.3,
      max_tokens: options.max_tokens || 512,
      temperature: options.temperature || 0.7,
    });

    return response.data;
  },

  searchDocuments: async (query, documentIds = [], limit = 10) => {
    return documentAPI.searchDocuments(query, {
      limit,
      document_ids: documentIds,
      top_k: limit,
      score_threshold: 0.3
    });
  },

  askQuestionWithContext: async (question, conversationContext = [], documentIds = [], options = {}) => {
    // FIX: Send a flat body matching the updated QueryRequest Pydantic model.
    // The old format { request: {...}, conversation_context: [...] } caused
    // FastAPI to 422 because `question` was missing at the top level.
    // That meant save_query_to_history() never ran → nothing persisted to DB
    // → all queries vanished on page reload.
    const requestBody = {
      question,
      document_ids: documentIds,
      top_k: options.top_k || 5,
      score_threshold: options.score_threshold || 0.3,
      max_tokens: options.max_tokens || 512,
      temperature: options.temperature || 0.3,
      conversation_context: conversationContext,  // now a field inside QueryRequest
    };

    const response = await api.post('/query/ask-with-context', requestBody);
    return response.data;
  },

  getQueryHistory: async (limit = 50) => {
    const response = await api.get(`/query/history?limit=${limit}`);
    return response.data;
  },

  getChatSessions: async () => {
    const response = await api.get('/query/sessions');
    return response.data;
  },

  getChatSession: async (sessionId) => {
    const response = await api.get(`/query/sessions/${sessionId}`);
    return response.data;
  },

  deleteSession: async (sessionId) => {
    const response = await api.delete(`/query/sessions/${sessionId}`);
    return response.data;
  },
};

// Analytics API functions
export const analyticsAPI = {
  getStats: async () => {
    try {
      const response = await api.get('/analytics/stats');
      return response.data;
    } catch (error) {
      console.error('Analytics API - getStats error:', error);
      return {
        total_queries: 0,
        total_documents: 0,
        avg_response_time: 0.0,
        successful_queries: 0,
        failed_queries: 0,
        last_updated: new Date().toISOString(),
        top_llm_used: null
      };
    }
  },

  getPopularQuestions: async (limit = 10) => {
    try {
      const response = await api.get(`/analytics/popular-questions?limit=${limit}`);
      return response.data;
    } catch (error) {
      console.error('Analytics API - getPopularQuestions error:', error);
      return {
        questions: [],
        total_unique_questions: 0
      };
    }
  },

  getQueryTrends: async (days = 7) => {
    try {
      const response = await api.get(`/analytics/query-trends?days=${days}`);
      return response.data;
    } catch (error) {
      console.error('Analytics API - getQueryTrends error:', error);
      return {
        success: false,
        trends: []
      };
    }
  },

  getLLMUsage: async () => {
    try {
      const response = await api.get('/analytics/llm-usage');
      return response.data;
    } catch (error) {
      console.error('Analytics API - getLLMUsage error:', error);
      return {
        success: false,
        llm_usage: {},
        top_llm: null,
        top_llm_count: 0
      };
    }
  }
};

// Search API
export const searchAPI = {
  search: async (query, options = {}) => {
    return documentAPI.searchDocuments(query, options);
  },
  
  // 🚀 Unified JSON Semantic Search
  semanticSearch: async (query, options = {}) => {
    const response = await api.post('/documents/search', {
      query: query,
      top_k: options.topK || 5,
      score_threshold: options.scoreThreshold || 0.3
    });
    return response.data;
  },
};

// Utility functions
export const utils = {
  formatFileSize: (bytes) => {
    if (!bytes || bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },

  formatDate: (dateString) => {
    if (!dateString) return '—';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  },

  validateFile: (file) => {
    const maxSize = 50 * 1024 * 1024; // 50MB
    const allowedTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/msword',
      'text/plain',
      'text/html',
      'text/markdown',
      'text/x-markdown',
      'application/x-pdf',
    ];
    
    const allowedExtensions = ['.pdf', '.docx', '.doc', '.txt', '.html', '.md'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

    if (file.size > maxSize) {
      throw new Error(`File "${file.name}" is too large. Maximum size is 50MB.`);
    }

    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
      throw new Error(`File "${file.name}" has unsupported format. Supported: PDF, Word, Text, HTML, Markdown`);
    }

    return true;
  },

  debounce: (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  highlightText: (text, query) => {
    if (!query || !text) return text;
    const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`(${escapedQuery})`, 'gi');
    return text.replace(regex, '<mark class="bg-yellow-200 px-1 rounded">$1</mark>');
  },

  extractSnippet: (text, query, maxLength = 200) => {
    if (!query || !text) return text.substring(0, maxLength) + '...';
    
    const lowerText = text.toLowerCase();
    const lowerQuery = query.toLowerCase();
    const index = lowerText.indexOf(lowerQuery);
    
    if (index === -1) {
      return text.substring(0, maxLength) + '...';
    }
    
    const start = Math.max(0, index - 50);
    const end = Math.min(text.length, index + query.length + 150);
    
    let snippet = text.substring(start, end);
    if (start > 0) snippet = '...' + snippet;
    if (end < text.length) snippet = snippet + '...';
    
    return snippet;
  },
};

export default api;