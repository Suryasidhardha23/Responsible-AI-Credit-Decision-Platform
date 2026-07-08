"""
Accounts views: login, logout, register, dashboard.

Security notes:
  - register_view redirects to login after successful registration.
  - login_view shows an error message on failed auth (no 200 on failure).
  - All sensitive dashboard content is behind @login_required.
"""

import logging

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from .forms import CustomUserCreationForm

logger = logging.getLogger(__name__)


@login_required
def dashboard(request):
    return render(request, 'accounts/dashboard.html', {'user': request.user})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')

    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            logger.info("User '%s' logged in successfully.", username)
            next_url = request.GET.get('next', 'home')
            return redirect(next_url)
        else:
            logger.warning("Failed login attempt for username: '%s'", username)
            error = "Invalid username or password. Please try again."

    return render(request, 'accounts/login.html', {'error': error})


def logout_view(request):
    username = request.user.username if request.user.is_authenticated else 'anonymous'
    logout(request)
    logger.info("User '%s' logged out.", username)
    messages.success(request, "You have been signed out successfully.")
    return redirect('login')


def register_view(request):
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            logger.info(
                "New user registered: '%s' with role '%s'.",
                user.username, user.role
            )
            messages.success(
                request,
                f"Account created for {user.username}! Please sign in to continue."
            )
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'accounts/register.html', {'form': form})
