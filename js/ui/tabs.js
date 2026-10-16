export function initTabs(app) {
  const buttons = [...document.querySelectorAll(".step")];
  const descriptions = {
    data: ["Collect data", "A few words. A few examples. A vocabulary that's yours."],
    train: ["Train model", "Turn your recorded signs into a model that recognises you."],
    live: ["Detect & speak", "Your signs become words, one sentence at a time."],
  };
  function show(tab) {
    buttons.forEach((button) => {
      const active = button.dataset.tab === tab;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${tab}`));
    document.getElementById("workspaceTitle").textContent = descriptions[tab][0];
    document.getElementById("workspaceDescription").textContent = descriptions[tab][1];
    document.getElementById("workspace").dataset.view = tab;
    app.emit("tab", tab);
  }
  buttons.forEach((button, index) => {
    button.addEventListener("click", () => show(button.dataset.tab));
    button.addEventListener("keydown", (event) => {
      let next = index;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (index + 1) % buttons.length;
      else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (index - 1 + buttons.length) % buttons.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = buttons.length - 1;
      else return;
      event.preventDefault();
      show(buttons[next].dataset.tab);
      buttons[next].focus();
    });
  });
  app.showTab = show;
}
