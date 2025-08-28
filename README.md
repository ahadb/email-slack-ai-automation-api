# 🚀 AI-Powered Message Summarization API

Automatically summarize and categorize conversations from email and Slack using OpenAI's GPT-4. This FastAPI service processes message batches and provides intelligent insights by extracting decisions, issues, and reminders.

## ✨ Features

• **Intelligent Message Summarization** - Powered by OpenAI GPT-4
• **Automatic Categorization** - Extracts decisions, issues, and reminders
• **PostgreSQL Database Storage** - Integrated with Supabase for persistence
• **LangSmith Integration** - Comprehensive LLM monitoring and tracing
• **Rate Limiting & Validation** - Built-in protection against abuse
• **Comprehensive Logging** - Request tracking and error monitoring
• **Connection Pooling** - Reliable database connections
• **Production Ready** - Error handling, validation, and monitoring

## 🛠️ Tech Stack

• **Backend**: FastAPI + Python 3.8+
• **AI/ML**: OpenAI GPT-4 via LangChain
• **Database**: PostgreSQL (Supabase)
• **Monitoring**: LangSmith for LLM operations
• **Deployment**: Docker-ready with environment configuration

## 🎯 Use Cases

• **Team Meeting Summaries** - Automatically extract key points from discussions
• **Customer Support Analysis** - Identify common issues and decisions
• **Project Status Tracking** - Monitor progress through conversation analysis
• **Automated Report Generation** - Create structured summaries from chat logs
• **Slack Channel Insights** - Understand team communication patterns

## 🚀 Quick Start

### Prerequisites

• Python 3.8+
• OpenAI API key
• Supabase PostgreSQL database
• LangSmith API key (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/email-slack-ai-automation-api.git
   cd email-slack-ai-automation-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual values
   ```

5. **Run the API**
   ```bash
   uvicorn main:app --reload
   ```

## ⚙️ Configuration

Create a `.env` file with the following variables:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Supabase Database URL
SUPABASE_DB_URL=your_supabase_database_url_here

# LangSmith Configuration (Optional)
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=email-slack-automation
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

## 📡 API Endpoints

### POST `/summarize`
Summarize a batch of messages and extract insights.

**Request Body:**
```json
{
  "source": "slack-channel",
  "messages": [
    {
      "sender": "john@example.com",
      "text": "Let's schedule a meeting for next week."
    },
    {
      "sender": "jane@example.com", 
      "text": "Great idea! How about Tuesday?"
    }
  ]
}
```

**Response:**
```json
{
  "summary": "Team discussed scheduling a meeting for next week, with Tuesday suggested as a potential date.",
  "categories": {
    "decisions": ["Schedule meeting for next week"],
    "issues": [],
    "reminders": ["Confirm meeting date"]
  },
  "saved": true,
  "request_id": "uuid-here",
  "processing_time": 2.34
}
```

### GET `/health`
Check API health status.

### GET `/traces`
Get recent LangSmith traces (requires LangSmith API key).

### GET `/logs`
Get LangSmith project information (requires LangSmith API key).

## 🗄️ Database Schema

The API expects the following tables in your Supabase PostgreSQL database:

```sql
-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(255) NOT NULL,
    sender VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Summaries table
CREATE TABLE summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_ids UUID[] NOT NULL,
    summary TEXT NOT NULL,
    categories JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🔒 Security & Rate Limiting

• **Rate Limiting**: 60 requests per minute per client
• **Input Validation**: Maximum 100 messages per request, 10,000 characters per message
• **Database Security**: Connection pooling with automatic cleanup
• **Error Handling**: Comprehensive logging without exposing sensitive information

## 📊 Monitoring & Observability

• **Request Tracking**: Unique request IDs for all operations
• **Performance Metrics**: LLM processing time tracking
• **LangSmith Integration**: Full LLM operation tracing and feedback
• **Structured Logging**: JSON-formatted logs with request context
• **Health Checks**: Database connectivity monitoring

## 🐳 Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🧪 Testing

```bash
# Run tests
python -m pytest

# Run with coverage
python -m pytest --cov=main
```

## 📈 Performance

• **Database**: Connection pooling for optimal performance
• **LLM**: Async processing with timeout handling
• **Caching**: Built-in rate limiting with memory storage
• **Monitoring**: Real-time performance tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

• **OpenAI** for GPT-4 API access
• **LangChain** for LLM orchestration
• **FastAPI** for the web framework
• **Supabase** for database hosting

## 📞 Support

• **Issues**: [GitHub Issues](https://github.com/yourusername/email-slack-ai-automation-api/issues)
• **Discussions**: [GitHub Discussions](https://github.com/yourusername/email-slack-ai-automation-api/discussions)
• **Documentation**: [API Docs](http://localhost:8000/docs) (when running locally)

---

**Made with ❤️ for teams who want to extract insights from their conversations automatically.**
