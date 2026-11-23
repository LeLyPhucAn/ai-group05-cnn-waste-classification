// script.js - Xử lý giao diện phía trình duyệt (ẩn phần tử thanh công cụ dư thừa)
(function () {
  const hideChrome = () => {
    try {
      const doc = window.parent.document;
      const selectors = [
        'header[data-testid="stHeader"]',
        '[data-testid="stToolbar"]',
        '[data-testid="stStatusWidget"]',
        '.stAppDeployButton',
        '.stDeployButton',
      ];
      selectors.forEach((sel) => {
        doc.querySelectorAll(sel).forEach((el) => {
          el.style.display = "none";
        });
      });
    } catch (e) {
      // Bỏ qua nếu không truy cập được parent document
    }
  };

  hideChrome();
  new MutationObserver(hideChrome).observe(window.parent.document.body, {
    childList: true,
    subtree: true,
  });
})();
