// Initialize tooltips
var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
});

// Initialize progress bars
document.addEventListener('DOMContentLoaded', function() {
    // Set width for all progress bars with data-width attribute
    document.querySelectorAll('.progress-bar[data-width]').forEach(progressBar => {
        const width = progressBar.getAttribute('data-width');
        progressBar.style.width = `${width}%`;
    });

    // Initialize any charts if the function exists
    if (typeof initCharts === 'function') {
        initCharts();
    }
});

// Initialize popovers
var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
    return new bootstrap.Popover(popoverTriggerEl);
});

// Handle sidebar toggle
const sidebarToggle = document.getElementById('sidebarToggle');
if (sidebarToggle) {
    sidebarToggle.addEventListener('click', function(e) {
        e.preventDefault();
        document.body.classList.toggle('sb-sidenav-toggled');
        localStorage.setItem('sb|sidebar-toggle', document.body.classList.contains('sb-sidenav-toggled'));
    });
}

// Handle active navigation links
const currentPath = window.location.pathname;
document.querySelectorAll('.nav-link').forEach(link => {
    if (link.getAttribute('href') === currentPath) {
        link.classList.add('active');
        // Also activate parent dropdown if exists
        const parentDropdown = link.closest('.dropdown');
        if (parentDropdown) {
            parentDropdown.querySelector('.dropdown-toggle').classList.add('active');
        }
    }
});

// Handle form submissions with loading state
const forms = document.querySelectorAll('.needs-validation');
Array.from(forms).forEach(form => {
    form.addEventListener('submit', function(event) {
        if (!form.checkValidity()) {
            event.preventDefault();
            event.stopPropagation();
        } else {
            const submitButton = form.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            }
        }
        form.classList.add('was-validated');
    }, false);
});

// Handle data table search
const searchInput = document.getElementById('searchInput');
if (searchInput) {
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        const tableBody = document.querySelector('tbody');
        if (tableBody) {
            const rows = tableBody.querySelectorAll('tr');
            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(searchTerm) ? '' : 'none';
            });
        }
    });
}

// Handle export modal form
const exportModal = document.getElementById('exportModal');
if (exportModal) {
    exportModal.addEventListener('show.bs.modal', function (event) {
        const button = event.relatedTarget;
        const format = button.getAttribute('data-format') || 'csv';
        const scope = button.getAttribute('data-scope') || 'all';
        
        const modal = this;
        modal.querySelector('#exportFormat').value = format;
        modal.querySelector('#exportScope').value = scope;
        
        // Toggle department select based on scope
        const deptSelectContainer = modal.querySelector('#departmentSelectContainer');
        if (deptSelectContainer) {
            deptSelectContainer.style.display = scope === 'department' ? 'block' : 'none';
        }
    });
    
    // Handle scope change
    const exportScope = document.getElementById('exportScope');
    if (exportScope) {
        exportScope.addEventListener('change', function() {
            const deptSelectContainer = document.getElementById('departmentSelectContainer');
            if (deptSelectContainer) {
                deptSelectContainer.style.display = this.value === 'department' ? 'block' : 'none';
            }
        });
    }
}

// Handle export button click
document.addEventListener('DOMContentLoaded', function() {
    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', function() {
            const format = document.getElementById('exportFormat').value;
            const scope = document.getElementById('exportScope').value;
            let url = `/export?format=${format}&scope=${scope}`;
            
            if (scope === 'department') {
                const department = document.getElementById('exportDepartment').value;
                if (department) {
                    url += `&department=${encodeURIComponent(department)}`;
                }
            }
            
            // In a real app, this would trigger a file download
            console.log('Export URL:', url);
            window.open(url, '_blank');
            
            // Close the modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('exportModal'));
            modal.hide();
        });
    }
});

// Handle student row click
document.querySelectorAll('.student-row').forEach(row => {
    row.addEventListener('click', function() {
        const studentId = this.getAttribute('data-id');
        // In a real app, this would navigate to the student's detailed view
        console.log('Viewing student:', studentId);
        window.location.href = `/student/${studentId}`;
    });
});

// Handle what-if scenario cards
document.querySelectorAll('.what-if-card').forEach(card => {
    card.addEventListener('click', function() {
        const scenarioId = this.getAttribute('data-scenario-id');
        // In a real app, this would show more details about the scenario
        console.log('Viewing scenario:', scenarioId);
    });
});

// Handle chart resizing on window resize
let resizeTimer;
window.addEventListener('resize', function() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function() {
        // Redraw charts here if needed
        if (typeof updateCharts === 'function') {
            updateCharts();
        }
    }, 250);
});

// Function to update charts (to be implemented by specific pages)
function updateCharts() {
    console.log('Updating charts...');
    // Chart update logic will be added by individual pages
}

// Handle password visibility toggle
const togglePassword = document.querySelector('.toggle-password');
if (togglePassword) {
    togglePassword.addEventListener('click', function() {
        const passwordInput = document.querySelector(this.getAttribute('toggle'));
        if (passwordInput) {
            const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordInput.setAttribute('type', type);
            this.querySelector('i').classList.toggle('fa-eye');
            this.querySelector('i').classList.toggle('fa-eye-slash');
        }
    });
}

// Handle file input change
document.querySelectorAll('.custom-file-input').forEach(input => {
    input.addEventListener('change', function() {
        const fileName = this.files[0]?.name || 'Choose file';
        const label = this.nextElementSibling;
        if (label) {
            label.textContent = fileName;
        }
    });
});

// Handle form validation
(function () {
    'use strict';
    window.addEventListener('load', function () {
        // Fetch all the forms we want to apply custom Bootstrap validation styles to
        var forms = document.getElementsByClassName('needs-validation');
        // Loop over them and prevent submission
        var validation = Array.prototype.filter.call(forms, function (form) {
            form.addEventListener('submit', function (event) {
                if (form.checkValidity() === false) {
                    event.preventDefault();
                    event.stopPropagation();
                }
                form.classList.add('was-validated');
            }, false);
        });
    }, false);
})();

// Handle AJAX form submissions
document.querySelectorAll('.ajax-form').forEach(form => {
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const submitButton = this.querySelector('button[type="submit"]');
        const originalButtonText = submitButton ? submitButton.innerHTML : '';
        
        // Show loading state
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
        }
        
        // Get the form's action and method
        const action = this.getAttribute('action') || window.location.href;
        const method = this.getAttribute('method') || 'POST';
        
        // Send the form data via fetch
        fetch(action, {
            method: method,
            body: formData,
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        })
        .then(response => response.json())
        .then(data => {
            // Handle success
            if (data.redirect) {
                window.location.href = data.redirect;
            } else if (data.message) {
                showAlert('success', data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('danger', 'An error occurred. Please try again.');
        })
        .finally(() => {
            // Reset button state
            if (submitButton) {
                submitButton.disabled = false;
                submitButton.innerHTML = originalButtonText;
            }
        });
    });
});

// Show alert message
function showAlert(type, message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    const container = document.querySelector('.alerts-container') || document.body;
    container.prepend(alertDiv);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alert = bootstrap.Alert.getOrCreateInstance(alertDiv);
        alert.close();
    }, 5000);
}

// Initialize any charts on the page
document.addEventListener('DOMContentLoaded', function() {
    if (typeof initCharts === 'function') {
        initCharts();
    }
});
