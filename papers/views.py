import json
import logging
from functools import wraps

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET

from .models import Paper, Conference, UserSession
from .facade import SearchFacade, IngestionFacade

logger = logging.getLogger(__name__)

search_facade    = SearchFacade()
ingestion_facade = IngestionFacade()


# ─── Auth decorator ───────────────────────────────────────────

def admin_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.session.get('is_admin'):
            return redirect('login')
        return view_func(request, *args, **kwargs)
    return wrapper


# ─── Public views ─────────────────────────────────────────────

def index(request):
    """Main search page."""
    conferences = Conference.objects.order_by('name')
    years = Paper.objects.values_list('year', flat=True).distinct().order_by('-year')
    return render(request, 'papers/index.html', {
        'conferences': conferences,
        'years': years,
    })


def search_api(request):
    """
    GET /api/search/?q=...&mode=ranked&conference=NeurIPS&year=2023&n=5
    Returns JSON for the frontend fetch() call.
    """
    query      = request.GET.get('q', '').strip()
    mode       = request.GET.get('mode', 'ranked')
    conference = request.GET.get('conference', '')
    year       = request.GET.get('year', '')
    n_results  = int(request.GET.get('n', 5))

    if not query:
        return JsonResponse({'error': 'Query is required.'}, status=400)

    result = search_facade.search(
        query=query,
        mode=mode,
        conference=conference,
        year=year,
        n_results=n_results,
    )
    return JsonResponse(result)


# ─── Auth views ───────────────────────────────────────────────

def login_view(request):
    if request.method == 'GET':
        if request.session.get('is_admin'):
            return redirect('dashboard')
        return render(request, 'papers/admin_login.html')

    # POST — validate credentials
    username = request.POST.get('username', '').strip()
    password = request.POST.get('password', '')

    user = authenticate(request, username=username, password=password)
    if user is None:
        return render(request, 'papers/admin_login.html', {'error': True})

    # Check admin subtype
    if not hasattr(user, 'admin_profile'):
        return render(request, 'papers/admin_login.html', {
            'error': True,
            'error_msg': 'This account does not have admin privileges.',
        })

    auth_login(request, user)
    request.session['is_admin']  = True
    request.session['admin_user'] = user.username
    request.session['full_name']  = f"{user.first_name} {user.last_name}"

    # Record session
    UserSession.objects.create(
        session_id   = request.session.session_key or 'unknown',
        user         = user,
    )
    return redirect('dashboard')


def logout_view(request):
    auth_logout(request)
    request.session.flush()
    return redirect('login')


# ─── Admin dashboard views ────────────────────────────────────

@admin_required
def dashboard(request):
    """Admin dashboard page."""
    total_papers     = Paper.objects.count()
    total_conferences = Conference.objects.count()
    indexed_papers   = Paper.objects.filter(indexed=True).count()
    recent_papers    = Paper.objects.select_related('conference').order_by('-added_at')[:10]

    return render(request, 'papers/admin_dashboard.html', {
        'total_papers'      : total_papers,
        'total_conferences' : total_conferences,
        'indexed_papers'    : indexed_papers,
        'recent_papers'     : recent_papers,
        'admin_username'    : request.session.get('admin_user', 'admin'),
        'full_name'         : request.session.get('full_name', 'Administrator'),
    })


@admin_required
@require_POST
def upload_file(request):
    """Handle CSV / JSON / BibTeX file upload via AJAX."""
    file = request.FILES.get('file')
    if not file:
        return JsonResponse({'error': 'No file provided.'}, status=400)

    ext = file.name.rsplit('.', 1)[-1].lower()
    if ext == 'csv':
        result = ingestion_facade.ingest_csv(file)
    elif ext == 'json':
        result = ingestion_facade.ingest_json(file)
    elif ext in ('bib', 'bibtex'):
        result = ingestion_facade.ingest_bibtex(file)
    else:
        return JsonResponse({'error': f'Unsupported file type: .{ext}'}, status=400)

    return JsonResponse(result)


@admin_required
@require_POST
def add_paper(request):
    """Handle manual paper entry form via AJAX."""
    data = {
        'title'      : request.POST.get('title', ''),
        'authors'    : request.POST.get('authors', ''),
        'abstract'   : request.POST.get('abstract', ''),
        'conference' : request.POST.get('conference', ''),
        'year'       : request.POST.get('year', ''),
        'doi_url'    : request.POST.get('doi_url', ''),
    }
    result = ingestion_facade.ingest_manual(data)
    return JsonResponse(result)


@admin_required
@require_POST
def delete_paper(request, paper_id):
    """Delete a paper by ID and remove from ChromaDB."""
    try:
        paper = Paper.objects.get(id=paper_id)

        # Remove from ChromaDB
        try:
            from .facade import _get_chroma_collection
            col = _get_chroma_collection()
            col.delete(ids=[str(paper_id)])
        except Exception as e:
            logger.warning("ChromaDB delete failed: %s", e)

        paper.delete()
        return JsonResponse({'success': True})
    except Paper.DoesNotExist:
        return JsonResponse({'error': 'Paper not found.'}, status=404)


@admin_required
def papers_list_api(request):
    """Return recent papers as JSON for dynamic table refresh."""
    papers = Paper.objects.select_related('conference').order_by('-added_at')[:20]
    data = [{
        'id'        : p.id,
        'title'     : p.title,
        'authors'   : p.authors,
        'conference': p.conference.name,
        'year'      : p.year,
        'indexed'   : p.indexed,
        'added_at'  : p.added_at.strftime('%d %b %Y'),
    } for p in papers]
    return JsonResponse({'papers': data})


@admin_required
def stats_api(request):
    """Return dashboard stats as JSON."""
    return JsonResponse({
        'total_papers'      : Paper.objects.count(),
        'total_conferences' : Conference.objects.count(),
        'indexed_papers'    : Paper.objects.filter(indexed=True).count(),
    })