
document.querySelectorAll('.collapsible-header').forEach(button => {
    button.addEventListener('click', function() {
        const content = this.nextElementSibling;

        // Toggle visibility
        content.style.display = content.style.display === 'block' ? 'none' : 'block';

        // Optionally, toggle the button text or icon

    });
});





// script.js
const carousel = document.querySelector('.carousel');
const leftBtn = document.querySelector('.left-btn');
const rightBtn = document.querySelector('.right-btn');

let scrollAmount = 0;

rightBtn.addEventListener('click', () => {
  const cardWidth = document.querySelector('.card').offsetWidth + 20; // 20px gap
  scrollAmount += cardWidth;
  carousel.style.transform = `translateX(-${scrollAmount}px)`;
});

leftBtn.addEventListener('click', () => {
  const cardWidth = document.querySelector('.card').offsetWidth + 20; // 20px gap
  scrollAmount -= cardWidth;
  if (scrollAmount < 0) scrollAmount = 0; // Prevent overscroll
  carousel.style.transform = `translateX(-${scrollAmount}px)`;
});



function displayFileName() {
  const fileInput = document.getElementById("file");
  const fileNameDisplay = document.getElementById("file-name");
  const fileName = fileInput.files[0]?.name || "لم يتم اختيار ملف.";
  fileNameDisplay.textContent = fileName;
}


