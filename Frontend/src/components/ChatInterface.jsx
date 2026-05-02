import React, { useState, useRef, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { queryAPI } from '../services/api';
import { 
  Send, 
  Bot, 
  User, 
  Sparkles, 
  Zap, 
  FileText,
  Brain,
  MessageSquare,
  ChevronRight,
  Loader,
  AlertCircle,
  Clock,
  Database,
  Cpu,
  TrendingUp,
  BookOpen,
  Search,
  Plus,
  History,
  X,
  MessageCircle,
  PanelLeftOpen,
  PanelLeftClose,
  Trash2
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import toast from 'react-hot-toast';
import { useAppStore } from '../store/store';
import websocketService from '../services/websocket';

// ─── Typing Indicator ───────────────────────────────────────────────────────
const TypingIndicator = () => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -10 }}
    className="flex items-center gap-2 text-gray-400"
  >
    <Bot className="w-5 h-5" />
    <div className="flex gap-1">
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          animate={{ y: [0, -10, 0] }}
          transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.2 }}
          className="w-2 h-2 bg-primary-400 rounded-full"
        />
      ))}
    </div>
    <span className="text-sm">AI is thinking...</span>
  </motion.div>
);

// ─── Streaming Toggle ────────────────────────────────────────────────────────
const StreamingToggle = ({ useStreaming, toggleStreaming }) => (
  <motion.button
    type="button"
    whileHover={{ scale: 1.05 }}
    whileTap={{ scale: 0.95 }}
    onClick={toggleStreaming}
    className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
    title={useStreaming ? 'Streaming enabled' : 'Streaming disabled'}
  >
    <Zap className={`w-4 h-4 ${useStreaming ? 'text-yellow-400' : 'text-gray-400'}`} />
    <span className="text-xs text-gray-300">
      {useStreaming ? 'Streaming' : 'Standard'}
    </span>
  </motion.button>
);

// ─── Message Component ───────────────────────────────────────────────────────
const MessageComponent = ({ message, index }) => {
  const isUser = message.role === 'user';
  const isCurrentlyStreaming = message.isStreaming;

  return (
    <motion.div
      initial={{ opacity: 0, x: isUser ? 20 : -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: 0 }}
      className={`flex gap-4 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
    >
      <motion.div
        whileHover={{ scale: 1.1 }}
        className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center shadow-lg
          ${isUser
            ? 'bg-gradient-to-r from-primary-500 to-purple-500'
            : 'bg-gradient-to-r from-green-500 to-emerald-500'
          }`}
      >
        {isUser ? <User className="w-5 h-5 text-white" /> : <Bot className="w-5 h-5 text-white" />}
      </motion.div>

      <div className={`flex-1 ${isUser ? 'text-right' : 'text-left'}`}>
        <div
         className={`inline-block max-w-3xl glass-card rounded-2xl p-4
         ${isUser ? 'bg-primary-500/10 border-primary-500/20' : ''}`}
        >
          {isUser ? (
            <p className="text-white">{message.content}</p>
          ) : (
            <div className="space-y-4">
              <div className="prose prose-invert max-w-none">
                <ReactMarkdown
                  components={{
                    p: ({ children }) => <p className="mb-2 text-gray-200">{children}</p>,
                    h1: ({ children }) => <h1 className="text-2xl font-bold mb-3 gradient-text">{children}</h1>,
                    h2: ({ children }) => <h2 className="text-xl font-bold mb-2 text-white">{children}</h2>,
                    h3: ({ children }) => <h3 className="text-lg font-semibold mb-2 text-white">{children}</h3>,
                    ul: ({ children }) => <ul className="list-disc list-inside space-y-1 mb-2">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal list-inside space-y-1 mb-2">{children}</ol>,
                    li: ({ children }) => <li className="text-gray-300">{children}</li>,
                    code: ({ inline, children }) =>
                      inline ? (
                        <code className="bg-white/10 px-1 py-0.5 rounded text-primary-300">{children}</code>
                      ) : (
                        <code className="block bg-black/30 p-3 rounded-lg overflow-x-auto text-gray-300">{children}</code>
                      ),
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-4 border-primary-400 pl-4 italic text-gray-400">
                        {children}
                      </blockquote>
                    ),
                  }}
                >
                  {message.content}
                </ReactMarkdown>

                {isCurrentlyStreaming && (
                  <motion.span
                    animate={{ opacity: [1, 0, 1] }}
                    transition={{ duration: 0.8, repeat: Infinity }}
                    className="inline-block w-1 h-4 bg-primary-400 ml-1"
                  />
                )}
              </div>
            {/*
              {message.sources && message.sources.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="mt-4 pt-4 border-t border-white/10"
                >
                  <p className="text-sm font-medium text-gray-400 mb-2 flex items-center gap-2">
                    <FileText className="w-4 h-4" />
                    Sources:
                  </p>
                  <div className="space-y-2">
                    {message.sources.map((source, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        whileHover={{ x: 5 }}
                        className="flex items-center gap-3 p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-all cursor-pointer"
                      >
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center text-white text-sm font-bold">
                          {idx + 1}
                        </div>
                        <div className="flex-1">
                          <p className="text-sm font-medium text-white">{source.document_name}</p>
                          <p className="text-xs text-gray-400">
                            Score: {(source.similarity_score * 100).toFixed(1)}% match
                          </p>
                        </div>
                        <ChevronRight className="w-4 h-4 text-gray-400" />
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}

              {message.metadata && (
                <div className="mt-3 flex items-center gap-4 text-xs text-gray-500">
                  {message.metadata.llm_used && (
                    <span className="flex items-center gap-1">
                      <Cpu className="w-3 h-3" />
                      {message.metadata.llm_used}
                    </span>
                  )}
                  {message.metadata.response_time && (
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {message.metadata.response_time.toFixed(2)}s
                    </span>
                  )}
                  {message.metadata.context_chunks_count && (
                    <span className="flex items-center gap-1">
                      <Database className="w-3 h-3" />
                      {message.metadata.context_chunks_count} chunks
                    </span>
                  )}
                </div>
              )}  */}
            </div>
          )}
        </div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-xs text-gray-500 mt-1 px-2"
        >
          {new Date(message.timestamp).toLocaleTimeString()}
        </motion.p>
      </div>
    </motion.div>
  );
};

// ─── Sessions Panel ──────────────────────────────────────────────────────────
const SessionsPanel = ({ isOpen, onClose, chatSessions, activeSessionId, onLoadSession, onNewChat, onDeleteSession }) => {
  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now - date;
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40"
          />

          {/* Slide Panel */}
          <motion.div
            initial={{ x: '-100%', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '-100%', opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed left-0 top-0 h-full w-80 z-50 flex flex-col"
            style={{
              background: 'linear-gradient(135deg, rgba(15,23,42,0.98) 0%, rgba(30,27,75,0.98) 100%)',
              borderRight: '1px solid rgba(255,255,255,0.08)',
              backdropFilter: 'blur(20px)',
            }}
          >
            {/* Panel Header */}
            <div className="flex items-center justify-between p-5 border-b border-white/10">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-primary-500 to-purple-500 flex items-center justify-center">
                  <History className="w-4 h-4 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold text-white text-sm">Chat History</h3>
                  <p className="text-xs text-gray-400">{chatSessions.length} sessions</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-all"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* New Chat Button */}
            <div className="p-4">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => { onNewChat(); onClose(); }}
                className="w-full flex items-center gap-3 px-4 py-3 rounded-xl
                  bg-gradient-to-r from-primary-500/20 to-purple-500/20
                  border border-primary-500/30 hover:border-primary-500/60
                  text-primary-300 hover:text-white transition-all group"
              >
                <div className="w-7 h-7 rounded-lg bg-gradient-to-r from-primary-500 to-purple-500 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Plus className="w-4 h-4 text-white" />
                </div>
                <span className="font-medium text-sm">New Chat</span>
              </motion.button>
            </div>

            {/* Sessions List */}
            <div className="flex-1 overflow-y-auto px-3 pb-4 space-y-1"
              style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,255,255,0.1) transparent' }}
            >
              {chatSessions.length === 0 ? (
                <div className="text-center py-12">
                  <MessageCircle className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                  <p className="text-gray-500 text-sm">No chat history yet</p>
                  <p className="text-gray-600 text-xs mt-1">Start a conversation!</p>
                </div>
              ) : (
                chatSessions.map((session, idx) => {
                  const isActive = session.id === activeSessionId;
                  return (
                    <motion.div
                      key={session.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.05 }}
                      className="relative group"
                    >
                      {/* Session load button */}
                      <motion.button
                        whileHover={{ x: 2 }}
                        onClick={() => { onLoadSession(session.id); onClose(); }}
                        className={`w-full text-left px-4 py-3 rounded-xl transition-all
                          ${isActive
                            ? 'bg-primary-500/20 border border-primary-500/40 text-white'
                            : 'hover:bg-white/5 border border-transparent hover:border-white/10 text-gray-300 hover:text-white'
                          }`}
                      >
                        <div className="flex items-start gap-3">
                          <div className={`mt-0.5 w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0
                            ${isActive
                              ? 'bg-gradient-to-r from-primary-500 to-purple-500'
                              : 'bg-white/10'
                            }`}
                          >
                            <MessageSquare className="w-3 h-3 text-white" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium truncate leading-tight">
                              {session.title}
                            </p>
                            <p className="text-xs text-gray-500 mt-0.5">
                              {formatDate(session.updated_at)} · {session.messages?.length || 0} messages
                            </p>
                          </div>
                          {isActive && (
                            <div className="w-2 h-2 rounded-full bg-primary-400 mt-1.5 flex-shrink-0" />
                          )}
                          {/* Delete button — inside, appears on hover */}
                          <motion.button
                            whileHover={{ scale: 1.15 }}
                            whileTap={{ scale: 0.9 }}
                            onClick={(e) => {
                              e.stopPropagation();
                              onDeleteSession(session.id);
                            }}
                          className="opacity-40 group-hover:opacity-100 flex-shrink-0 w-6 h-6 rounded-md flex items-center justify-center
                              hover:bg-red-500/25 text-gray-500 hover:text-red-400 transition-all duration-150"
                            title="Delete session"
                          >
                            <Trash2 className="w-3 h-3" />
                          </motion.button>
                        </div>
                      </motion.button>
                    </motion.div>
                  );
                })
              )}
            </div>

            {/* Panel Footer */}
            <div className="p-4 border-t border-white/10">
              <p className="text-xs text-gray-600 text-center">
                Sessions saved permanently · PostgreSQL
              </p>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

// ─── Main Chat Interface ─────────────────────────────────────────────────────
const ChatInterface = () => {
  const [input, setInput] = useState('');
  const [selectedDocumentIds, setSelectedDocumentIds] = useState([]);
  const [showDocDropdown, setShowDocDropdown] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const dropdownRef = useRef(null);
  const [suggestions, setSuggestions] = useState([]);
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [showSessionsPanel, setShowSessionsPanel] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const lastQuestionRef = useRef('');

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setShowDocDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const {
    queryHistory,
    askQuestion,
    askQuestionWithStreaming,
    isLoading,
    documents,
    selectedDocuments,
    setActiveTab,
    conversationContext,
    clearConversation,
    useStreaming,
    toggleStreaming,
    updateStreamingMessage,
    streamingMessageId,
    chatSessions,
    activeSessionId,
    loadSession,
    deleteSession,
  } = useAppStore();

  // Auto-select docs from Documents page when redirected
  useEffect(() => {
    if (selectedDocuments && selectedDocuments.length > 0) {
      setSelectedDocumentIds(selectedDocuments);
    }
  }, []);

  // Transform queryHistory to messages
  const messages = useMemo(() => {
    if (!queryHistory) return [];
    return [...queryHistory].flatMap(item => {
      const userMessage = {
        id: `${item.id}_user`,
        role: 'user',
        content: item.question || item.content || '',
        timestamp: item.timestamp
      };
      const assistantMessage = {
        id: `${item.id}_assistant`,
        role: 'assistant',
        content: item.answer?.answer || item.content || 'No response',
        sources: item.answer?.sources || item.sources || [],
        metadata: {
          response_time: item.answer?.response_time,
          llm_used: item.llm_used || 'Unknown',
          context_chunks_count: item.context_chunks_count
        },
        timestamp: item.timestamp,
        isStreaming: item.isStreaming,
        error: item.isError
      };
      return [userMessage, assistantMessage];
    });
  }, [queryHistory]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Suggestions while typing
  useEffect(() => {
    if (input.length > 2) {
      const historicalQuestions = queryHistory
        .map(item => item.question)
        .filter(q => q && q.toLowerCase().includes(input.toLowerCase()))
        .slice(0, 5);
      setSuggestions(historicalQuestions);
      setShowSuggestions(historicalQuestions.length > 0);
    } else {
      setShowSuggestions(false);
    }
  }, [input, queryHistory]);

  // WebSocket streaming listeners
  useEffect(() => {
    websocketService.on('answer_stream_start', () => {
      setIsStreaming(true);
      setStreamingAnswer('');
    });

    websocketService.on('answer_stream_chunk', (data) => {
      setStreamingAnswer(data.content);
      if (streamingMessageId) {
        updateStreamingMessage(streamingMessageId, data.content, false);
      }
    });

    websocketService.on('answer_stream_end', (data) => {
      setIsStreaming(false);
      if (streamingMessageId) {
        updateStreamingMessage(streamingMessageId, data.content, true);
        useAppStore.setState((state) => ({
          conversationContext: [
            ...state.conversationContext.slice(-2),
            { question: lastQuestionRef.current, answer: data.content }
          ],
          // Store session_id so next question in same chat uses same session
          activeSessionId: data.session_id || state.activeSessionId
        }));

        // Refresh sessions list in sidebar
        queryAPI.getChatSessions().then(sessions => {
          useAppStore.setState({ chatSessions: sessions });
        }).catch(() => {});

        // ── Live analytics refresh after every answer ──────────────────
        import('../store/store').then(({ useAnalyticsStore }) => {
          const s = useAnalyticsStore.getState();
          s.loadStats();
          s.loadQueryTrends();
        }).catch(() => {});
      }
    });

    websocketService.on('answer_stream_error', (data) => {
      setIsStreaming(false);
      toast.error('Streaming failed: ' + data.error);
    });

    return () => {
      websocketService.off('answer_stream_start');
      websocketService.off('answer_stream_chunk');
      websocketService.off('answer_stream_end');
      websocketService.off('answer_stream_error');
    };
  }, [streamingMessageId, updateStreamingMessage]);

  const sendMessage = async (question, documentIds) => {
    try {
      if (useStreaming && websocketService.ws?.readyState === WebSocket.OPEN) {
        await askQuestionWithStreaming(question, { document_ids: documentIds || [] });
      } else {
        await askQuestion(question, { document_ids: documentIds || [] });
      }
    } catch (error) {
      toast.error(error.message || 'Failed to send message.');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    const userMessage = input.trim();
    lastQuestionRef.current = userMessage;
    setInput('');
    try {
      await sendMessage(userMessage, selectedDocumentIds);
    } catch (error) {
      toast.error('Failed to send message. Please try again.');
    }
  };

  const handleNewChat = () => {
    clearConversation();
    setInput('');
    setStreamingAnswer('');
  };

  // Delete a session permanently + refresh analytics immediately
  const handleDeleteSession = async (sessionId) => {
    try {
      await deleteSession(sessionId);
      // If the active session was deleted, start a fresh chat
      if (sessionId === activeSessionId) {
        handleNewChat();
      }
      // Immediately refresh analytics so numbers update live
      try {
        const { useAnalyticsStore } = await import('../store/store');
        const analyticsStore = useAnalyticsStore.getState();
        analyticsStore.loadStats();
        analyticsStore.loadQueryTrends();
      } catch (_) {}
    } catch (err) {
      toast.error('Could not delete session. Please try again.');
    }
  };

  const suggestedQuestions = [
    { icon: BookOpen, text: 'Summarize the main points from my documents', color: 'from-blue-500 to-cyan-500' },
    { icon: Search, text: 'What are the key findings mentioned?', color: 'from-purple-500 to-pink-500' },
    { icon: Brain, text: 'Explain the technical concepts in simple terms', color: 'from-green-500 to-emerald-500' },
    { icon: TrendingUp, text: 'What trends are highlighted in the data?', color: 'from-orange-500 to-red-500' }
  ];

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] max-w-6xl mx-auto relative">

      {/* Sessions Slide Panel */}
      <SessionsPanel
        isOpen={showSessionsPanel}
        onClose={() => setShowSessionsPanel(false)}
        chatSessions={chatSessions || []}
        activeSessionId={activeSessionId}
        onLoadSession={loadSession}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
      />

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card rounded-2xl px-6 py-4 mb-4 overflow-visible"
      >
        <div className="flex items-center justify-between gap-6">

          {/* Left — History + Assistant Info */}
          <div className="flex items-center gap-4 flex-shrink-0">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowSessionsPanel(true)}
              className="relative p-2.5 rounded-xl bg-white/10 hover:bg-white/20 transition-all group"
              title="Chat History"
            >
              <PanelLeftOpen className="w-5 h-5 text-gray-400 group-hover:text-white transition-colors" />
              {chatSessions && chatSessions.length > 0 && (
                <span className="absolute -top-1 -right-1 w-4 h-4 bg-primary-500 rounded-full text-xs text-white flex items-center justify-center font-bold">
                  {chatSessions.length > 9 ? '9+' : chatSessions.length}
                </span>
              )}
            </motion.button>

            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-r from-primary-500 to-purple-500 flex items-center justify-center shadow-lg">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <motion.div
                  animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.6, 0.3] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="absolute inset-0 bg-primary-400/30 rounded-xl blur-lg"
                />
              </div>
              <div>
                <h2 className="text-base font-bold text-white">AI Assistant</h2>
                <p className="text-xs text-gray-400 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
                  {conversationContext.length > 0
                    ? `${conversationContext.length} messages in context`
                    : 'Ready to help with your documents'
                  }
                </p>
              </div>
            </div>
          </div>

          {/* Right — Controls */}
          <div className="flex items-center gap-3 flex-shrink-0 overflow-visible">
            {documents && documents.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">Search in:</span>
                <div className="relative" ref={dropdownRef}>
                  {/* Trigger Button */}
                  <button
                    type="button"
                    onClick={() => setShowDocDropdown(!showDocDropdown)}
                    className="flex items-center gap-2 bg-white/5 border border-white/10 text-slate-300 rounded-lg px-3 py-1.5 text-xs hover:bg-white/10 transition-all cursor-pointer min-w-[130px] max-w-[180px]"
                  >
                    <span className="flex-1 text-left truncate">
                      {selectedDocumentIds.length === 0
                        ? 'All documents'
                        : selectedDocumentIds.length === 1
                        ? documents.find(d => d.id === selectedDocumentIds[0])?.filename || '1 selected'
                        : `${selectedDocumentIds.length} docs selected`}
                    </span>
                    <svg className={`w-3 h-3 text-gray-500 transition-transform flex-shrink-0 ${showDocDropdown ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {/* Compact Dropdown */}
                  {showDocDropdown && (
                    <div
                      className="absolute right-0 top-8 w-52 z-[9999] rounded-lg border border-white/10 shadow-xl overflow-hidden"
                      style={{ background: 'rgba(17,24,39,0.97)', backdropFilter: 'blur(16px)' }}
                    >
                      {/* All Documents */}
                      <button
                        type="button"
                        onClick={() => setSelectedDocumentIds([])}
                        className={`w-full flex items-center gap-2 px-3 py-2 text-xs transition-all hover:bg-white/5 ${selectedDocumentIds.length === 0 ? 'text-primary-400' : 'text-gray-400'}`}
                      >
                        <div className={`w-3 h-3 rounded border flex items-center justify-center flex-shrink-0 transition-all ${selectedDocumentIds.length === 0 ? 'bg-primary-500 border-primary-500' : 'border-white/20'}`}>
                          {selectedDocumentIds.length === 0 && (
                            <svg className="w-2 h-2 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                        </div>
                        <span className="font-medium">All documents</span>
                      </button>

                      <div className="mx-3 border-t border-white/5" />

                      {/* Individual docs — 4 visible, then scroll */}
                      <div className="overflow-y-auto" style={{ maxHeight: '132px', scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,255,255,0.08) transparent' }}>
                        {documents.map((doc) => {
                          const isChecked = selectedDocumentIds.includes(doc.id);
                          return (
                            <button
                              type="button"
                              key={doc.id}
                              onClick={() => {
                                setSelectedDocumentIds(prev =>
                                  prev.includes(doc.id)
                                    ? prev.filter(id => id !== doc.id)
                                    : [...prev, doc.id]
                                );
                              }}
                              className={`w-full flex items-center gap-2 px-3 py-2 text-xs transition-all hover:bg-white/5 ${isChecked ? 'text-primary-300' : 'text-gray-400'}`}
                            >
                              <div className={`w-3 h-3 rounded border flex items-center justify-center flex-shrink-0 transition-all ${isChecked ? 'bg-primary-500 border-primary-500' : 'border-white/20'}`}>
                                {isChecked && (
                                  <svg className="w-2 h-2 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                                  </svg>
                                )}
                              </div>
                              <span className="truncate text-left">{doc.filename}</span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
            <StreamingToggle useStreaming={useStreaming} toggleStreaming={toggleStreaming} />
          </div>

        </div>
      </motion.div>
      <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2"
        style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,255,255,0.1) transparent' }}
      >
        {messages.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="h-full flex flex-col items-center justify-center text-center"
          >
            <motion.div
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 3, repeat: Infinity }}
              className="mb-8"
            >
              <MessageSquare className="w-20 h-20 text-primary-400/50 mx-auto" />
            </motion.div>

            <h3 className="text-2xl font-bold gradient-text mb-4">Start a Conversation</h3>
            <p className="text-gray-400 mb-8 max-w-md">
              Ask questions about your documents and get intelligent answers powered by advanced RAG technology
            </p>
      
          </motion.div>
        ) : (
          <>
            {messages.filter(m => {
              // Hide empty streaming placeholder messages
              if (m.role === 'assistant' && (!m.content || m.content === 'No response') && m.isStreaming) return false;
              return true;
            }).map((message, index) => (
              <MessageComponent key={message.id} message={message} index={index} />
            ))}

            <AnimatePresence>
              {isLoading && !isStreaming && <TypingIndicator />}
            </AnimatePresence>

            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input Form */}
      <motion.form
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        onSubmit={handleSubmit}
        className="glass-card rounded-2xl p-4"
      >
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <div className="relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                placeholder="Ask a question about your documents..."
                className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl
                         text-white placeholder-gray-400 resize-none
                         focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent
                         transition-all duration-300"
                rows={1}
                style={{ minHeight: '48px', maxHeight: '120px' }}
                onInput={(e) => {
                  e.target.style.height = 'auto';
                  e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
                }}
              />

              <AnimatePresence>
                {showSuggestions && suggestions.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="absolute bottom-full mb-2 left-0 right-0 bg-gray-900/95 backdrop-blur-md rounded-xl border border-white/10 overflow-hidden shadow-xl"
                  >
                    <div className="p-2">
                      <p className="text-xs text-gray-400 px-3 py-1 flex items-center gap-1">
                        <History className="w-3 h-3" />
                        Recent questions
                      </p>
                      {suggestions.map((suggestion, idx) => (
                        <button
                          type="button"
                          key={idx}
                          onClick={() => { setInput(suggestion); setShowSuggestions(false); }}
                          className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-white/10 rounded-lg transition-colors"
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {input.length > 0 && (
                <span className="absolute bottom-2 right-2 text-xs text-gray-500">
                  {input.length} / 1000
                </span>
              )}
            </div>

            <div className="flex items-center gap-2 mt-2">
              <span className="text-xs text-gray-500">Press Enter to send</span>
              {documents.length === 0 && (
                <span className="text-xs text-yellow-400 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" />
                  No documents uploaded yet
                </span>
              )}
            </div>
          </div>

          <motion.button
            type="submit"
            disabled={!input.trim() || isLoading}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="p-3 rounded-xl font-semibold text-white
              bg-gradient-to-r from-primary-500 to-purple-500
              hover:from-primary-600 hover:to-purple-600
              disabled:opacity-50 disabled:cursor-not-allowed
              transition-all duration-300
              shadow-lg hover:shadow-xl hover:shadow-primary-500/25
              flex items-center gap-2"
          >
            {isLoading ? (
              <Loader className="w-5 h-5 animate-spin" />
            ) : (
              <>
                <Send className="w-5 h-5" />
                <Sparkles className="w-4 h-4" />
              </>
            )}
          </motion.button>
        </div>
      </motion.form>
    </div>
  );
};

export default ChatInterface;
