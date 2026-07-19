import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageCircle, X, Send, Bot, User } from "lucide-react";

interface Message {
  role: "user" | "bot";
  text: string;
}

interface ChatWidgetProps {
  selectedSymbol?: string;
}

export default function ChatWidget({ selectedSymbol }: ChatWidgetProps) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    { role: "bot", text: "Hi! I'm AInvestify AI. Ask me about any stock, market trends, or investment strategies." },
  ]);
  const [typing, setTyping] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing]);

  const send = async () => {
    if (!input.trim()) return;
    const userMsg = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", text: userMsg }]);
    setTyping(true);
    try {
      const resp = await fetch("/chatbot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMsg,
          selected_symbol: selectedSymbol || null,
        }),
      });
      const payload = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        const errMsg =
          payload?.detail?.error?.message ||
          payload?.detail ||
          "Chat service unavailable right now.";
        throw new Error(errMsg);
      }
      const reply = String(payload?.reply || "").trim() || "No response from assistant.";
      setMessages(prev => [...prev, { role: "bot", text: reply }]);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Chat service unavailable right now.";
      setMessages(prev => [...prev, { role: "bot", text: message }]);
    } finally {
      setTyping(false);
    }
  };

  return (
    <>
      {/* Toggle Button */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setOpen(o => !o)}
        className="fixed bottom-6 right-6 w-12 h-12 rounded-full bg-primary flex items-center justify-center shadow-pop z-50 cursor-pointer"
      >
        {open ? <X className="w-5 h-5 text-primary-foreground" /> : <MessageCircle className="w-5 h-5 text-primary-foreground" />}
      </motion.button>

      {/* Chat Panel */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className="fixed bottom-20 right-6 w-80 sm:w-96 rounded-xl border border-border bg-card shadow-xl z-50 flex flex-col overflow-hidden"
            style={{ maxHeight: "min(500px, 70vh)" }}
          >
            {/* Header */}
            <div className="flex items-center gap-2 px-4 py-3 border-b border-border bg-secondary/30">
              <Bot className="w-5 h-5 text-primary" />
              <span className="text-sm font-semibold text-foreground">AInvestify AI</span>
              <span className="ml-auto w-2 h-2 rounded-full bg-success animate-pulse-soft" />
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3" style={{ minHeight: 200 }}>
              {messages.map((msg, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex gap-2 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
                >
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 ${msg.role === "bot" ? "bg-primary/10" : "bg-secondary"}`}>
                    {msg.role === "bot" ? <Bot className="w-3 h-3 text-primary" /> : <User className="w-3 h-3 text-muted-foreground" />}
                  </div>
                  <div className={`max-w-[80%] px-3 py-2 rounded-lg text-sm ${
                    msg.role === "user" ? "bg-primary text-primary-foreground" : "bg-secondary/80 text-foreground"
                  }`}>
                    {msg.text}
                  </div>
                </motion.div>
              ))}
              {typing && (
                <div className="flex gap-2">
                  <div className="w-6 h-6 rounded-full flex items-center justify-center bg-primary/10">
                    <Bot className="w-3 h-3 text-primary" />
                  </div>
                  <div className="px-3 py-2 rounded-lg bg-secondary/80">
                    <div className="flex gap-1">
                      {[0, 1, 2].map(i => (
                        <div key={i} className="w-1.5 h-1.5 rounded-full bg-muted-foreground animate-pulse-soft" style={{ animationDelay: `${i * 0.2}s` }} />
                      ))}
                    </div>
                  </div>
                </div>
              )}
              <div ref={endRef} />
            </div>

            {/* Input */}
            <div className="flex items-center gap-2 p-3 border-t border-border">
              <input
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && send()}
                placeholder="Ask about stocks..."
                className="flex-1 h-9 px-3 rounded-lg border border-border bg-secondary/50 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
              />
              <button onClick={send} className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center hover:bg-primary/90 transition-colors cursor-pointer">
                <Send className="w-4 h-4 text-primary-foreground" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
