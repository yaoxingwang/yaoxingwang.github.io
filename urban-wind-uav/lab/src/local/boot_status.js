// Loading hint for static hosting (build_lab.py): main.js shows the loading card only once all of its modules have
// arrived. Until then this shows the card with a download message, says so when that takes long, and asks for a
// reload when a module failed to download. Display only; it never touches the computation.
const DOWNLOADING = "正在下载程序文件…";
const SLOW = "下载时间较长：网络可能较慢，或有文件没下载成功。可以继续等待，或刷新页面重试。";
const FAILED = "有文件没下载成功（网络不稳定）。请刷新页面重试。";
const card = document.getElementById("loading-card");
const text = document.getElementById("loading-text");
const ours = () => text.textContent === DOWNLOADING || text.textContent === SLOW;   // main.js has not taken over
card.classList.add("show");
text.textContent = DOWNLOADING;
setTimeout(() => { if (ours()) text.textContent = SLOW; }, 30000);
document.querySelector('script[src="src/main.js"]')?.addEventListener("error", () => {
  text.textContent = FAILED;
  const spinner = card.querySelector(".spinner");
  if (spinner) spinner.style.display = "none";
});
