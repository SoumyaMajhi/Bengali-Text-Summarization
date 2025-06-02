document.addEventListener("DOMContentLoaded", () => {
  const inputText = document.getElementById("inputText");
  const summaryText = document.getElementById("summaryText");
  const wordCount = document.getElementById("wordCount");
  const summarizeBtn = document.getElementById("summarizeBtn");
  const clearBtn = document.getElementById("clearBtn");
  const spinner = document.getElementById("spinner");
  const darkModeToggle = document.getElementById("darkModeToggle");
const alertPlaceholder = document.getElementById('alertPlaceholder');


  // Function to show alert
const showAlert = (message) => {
  alertPlaceholder.innerHTML = `
    <div class="alert alert-warning alert-dismissible fade show mt-3 w-75 mx-auto" role="alert">
      <strong>⚠️ ওহ!</strong> ${message}
      
    </div>
  `;

  // Optional: Auto-dismiss after 5 seconds
  setTimeout(() => {
    const alert = document.querySelector('.alert');
    if (alert) {
      alert.classList.remove('show');
      alert.classList.add('fade');
      setTimeout(() => alert.remove(), 300); // Remove from DOM after fade out
    }
  }, 2000);
};


  // Update word count on input
  inputText.addEventListener("input", () => {
    const words = inputText.value.trim().split(/\s+/).filter(Boolean);
    wordCount.textContent = words.length;
  });

  // Summarize button click handler
  summarizeBtn.addEventListener("click", async () => {
    const text = inputText.value.trim();
    if (!text) {
      showAlert("দয়া করে একটি টেক্সট লিখুন!");
      return;
    }

    summaryText.textContent = "";
    spinner.classList.remove("d-none");

    try {
      const response = await fetch("/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ original_text: text }),
      });

      if (!response.ok) throw new Error("সার্ভার সমস্যা!");

      const data = await response.json();
      summaryText.textContent = data.summary || "❌ সংক্ষিপ্তকরণ ব্যর্থ হয়েছে";
    } catch (err) {
      summaryText.textContent = "❌ সংযোগ সমস্যার কারণে সংক্ষিপ্তকরণ ব্যর্থ হয়েছে";
      console.error(err);
    } finally {
      spinner.classList.add("d-none");
    }
  });

  // Clear button click handler
  clearBtn.addEventListener("click", () => {
    inputText.value = "";
    summaryText.textContent = "সংক্ষিপ্ত টেক্সট এখানে দেখানো হবে";
    wordCount.textContent = "0";
  });

  // Load dark mode preference from localStorage
  // if (localStorage.getItem("darkMode") === "enabled") {
  //   document.body.classList.add("dark-mode");
  //   darkModeToggle.checked = true;
  // }

  // // Dark mode toggle handler with persistence
  // darkModeToggle.addEventListener("change", () => {
  //   document.body.classList.toggle("dark-mode");
  //   if (document.body.classList.contains("dark-mode")) {
  //     localStorage.setItem("darkMode", "enabled");
  //   } else {
  //     localStorage.setItem("darkMode", "disabled");
  //   }
  // });

  const themeToggle = document.querySelector('.theme-switch__checkbox');

  // Load saved theme preference
  if (localStorage.getItem('theme') === 'dark') {
    document.body.classList.add('dark-mode');
    themeToggle.checked = true;
  }

  themeToggle.addEventListener('change', () => {
    document.body.classList.toggle('dark-mode');

    if (document.body.classList.contains('dark-mode')) {
      localStorage.setItem('theme', 'dark');
    } else {
      localStorage.setItem('theme', 'light');
    }
  });
});
