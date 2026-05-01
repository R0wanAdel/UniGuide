const askBtn = document.getElementById('askBtn');
const questionInput = document.getElementById('question');
const answerBox = document.getElementById('answer-box');
const sourcesBox = document.getElementById('sources-box');
const faqButtons = document.querySelectorAll('.faq-btn');

// تشغيل الأسئلة الجاهزة
faqButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        questionInput.value = btn.getAttribute('data-q');
        askQuestion();
    });
});

// الدالة الأساسية (لازم يكون قبلها كلمة async عشان await تشتغل جوه)
async function askQuestion() {
    const question = questionInput.value.trim();

    if (!question) {
        answerBox.innerHTML = `<div class="answer-card">من فضلك اكتب سؤالك.</div>`;
        sourcesBox.innerHTML = '';
        return;
    }

    // إظهار رسالة التحميل
    answerBox.innerHTML = `<div class="answer-card">جاري البحث...</div>`;
    sourcesBox.innerHTML = '';

    try {
        // الـ await هنا بتشتغل عادي لأن الدالة فوق مكتوب قبلها async
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                top_k: 3
            })
        });

        const data = await response.json();

        // عرض الإجابة
        answerBox.innerHTML = `<div class="answer-card">
            <span style="font-size: 24px;">🤖</span><br><br>
            <b>بناءً على اللائحة:</b><br>
            ${data.answer || data.final_answer || 'لا توجد إجابة حالياً.'}
        </div>`;

        // عرض المصادر لو موجودة
        let sourcesHTML = '';
        const sources = data.sources || data.chunks_used;

        if (Array.isArray(sources) && sources.length) {
            sources.forEach(src => {
                sourcesHTML += `
                    <div class="source-card">
                        <strong>${src.title || src.article || 'المصدر'}</strong><br><br>
                        ${src.content || src.text || ''}
                    </div>`;
            });
        } else {
            sourcesHTML = `<div class="source-card">لا توجد مصادر محددة.</div>`;
        }

        sourcesBox.innerHTML = sourcesHTML;

    } catch (err) {
        console.error("Fetch Error:", err);
        answerBox.innerHTML = `<div class="answer-card" style="color: red;">
            حدث خطأ أثناء جلب الإجابة. تأكدي من تشغيل السيرفر (FastAPI).
        </div>`;
    }
}

// ربط زرار البحث بالدالة
askBtn.addEventListener('click', askQuestion);

// تشغيل البحث لما اليوزر يدوس Enter
questionInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') askQuestion();
});