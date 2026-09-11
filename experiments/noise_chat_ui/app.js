const transcript = document.querySelector('#transcript');
const form = document.querySelector('#chat-form');
const input = document.querySelector('#message');
const send = document.querySelector('#send');
let loaded = false;

const intentLabels = {
  greeting: '挨拶', claim: '説明', ask_meaning: '意味の質問',
  open_question: '未解決の質問', elliptical_question: '例を尋ねる質問',
  recall_knowledge: '記憶の確認', confirm_understanding: '理解の確認',
  ask_curiosity: '好奇心の確認', correction: '訂正', parse_feedback: '解析への訂正',
  preference: 'あなたの好み', recall_preference: '好みの確認',
  learning_status: '学習状況', activity_status: '現在の活動',
  ask_noise_preference: 'Noiseの好み', unresolved: 'まだ解釈できない発話',
};

function interpretationText(read) {
  if (!read || !read.intent) return '';
  const kind = intentLabels[read.intent] || read.intent;
  const topic = read.topic ? ` ・話題「${read.topic}」` : '';
  return `Noiseの読み: ${kind}${topic}`;
}

function message(role, text, pending = false, read = null) {
  const article = document.createElement('article');
  article.className = `message ${role}${pending ? ' pending' : ''}`;
  const speaker = document.createElement('div');
  speaker.className = 'speaker';
  speaker.textContent = role === 'user' ? 'YOU' : 'NOISE';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  article.append(speaker, bubble);
  const explanation = interpretationText(read);
  if (role === 'noise' && explanation) {
    const meta = document.createElement('div');
    meta.className = 'interpretation';
    meta.textContent = explanation;
    article.append(meta);
  }
  transcript.append(article);
  transcript.scrollTop = transcript.scrollHeight;
  return article;
}

function renderTurns(turns) {
  if (loaded) return;
  if (turns.length) transcript.replaceChildren();
  for (const turn of turns) {
    message('user', turn.user || '');
    message('noise', turn.noise || '', false, turn.interpretation || null);
  }
  loaded = true;
}

function renderState(state) {
  const convo = state.conversation || {};
  const worker = state.worker || {};
  document.querySelector('#turns').textContent = Number(convo.turns || 0).toLocaleString('ja-JP');
  document.querySelector('#memories').textContent = Number(convo.remembered_subjects || 0).toLocaleString('ja-JP');
  document.querySelector('#corrections').textContent = Number(convo.corrections || 0).toLocaleString('ja-JP');
  document.querySelector('#unknown').textContent = Number(convo.unknown_topics || 0).toLocaleString('ja-JP');
  document.querySelector('#worker-state').textContent = worker.alive ? '稼働中' : '停止';
  document.querySelector('#worker-phase').textContent = worker.phase_ja || '不明';
  document.querySelector('#worker-seed').textContent = worker.seed || '—';
  const liveDot = document.querySelector('#live-dot');
  liveDot.classList.toggle('ok', Boolean(worker.alive));
  document.querySelector('#live-label').textContent = worker.alive ? '自動学習と接続' : '会話のみ利用可能';
  const topics = document.querySelector('#topic-list');
  topics.replaceChildren();
  const values = convo.unknown_topics_recent || [];
  for (const value of values.length ? values : ['まだありません']) {
    const tag = document.createElement('span');
    tag.textContent = value;
    topics.append(tag);
  }
  renderTurns(convo.turns_recent || []);
}

async function refresh() {
  try {
    const response = await fetch('/api/state', {cache: 'no-store'});
    if (!response.ok) throw new Error('state request failed');
    renderState(await response.json());
  } catch (_) {
    document.querySelector('#live-label').textContent = 'GUI接続エラー';
    document.querySelector('#live-dot').classList.remove('ok');
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || send.disabled) return;
  message('user', text);
  input.value = '';
  input.style.height = 'auto';
  send.disabled = true;
  const waiting = message('noise', '考えています…', true);
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '送信に失敗しました');
    waiting.querySelector('.bubble').textContent = result.reply;
    waiting.classList.remove('pending');
    const read = result.state?.conversation?.last_exchange?.interpretation;
    const explanation = interpretationText(read);
    if (explanation) {
      const meta = document.createElement('div');
      meta.className = 'interpretation';
      meta.textContent = explanation;
      waiting.append(meta);
    }
    renderState(result.state);
  } catch (error) {
    waiting.querySelector('.bubble').textContent = `会話できませんでした: ${error.message}`;
    waiting.classList.remove('pending');
  } finally {
    send.disabled = false;
    input.focus();
  }
});

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});

input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll('[data-prompt]').forEach((button) => {
  button.addEventListener('click', () => {
    input.value = button.dataset.prompt;
    input.focus();
  });
});

refresh();
setInterval(refresh, 10000);
